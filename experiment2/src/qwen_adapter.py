"""Full-image FP16 Qwen inference with T4-compatible efficient attention."""
from contextlib import ExitStack
from time import perf_counter

from .attention import efficient_attention, validate_efficient_attention
import json
from pathlib import Path

from PIL import Image


class QwenImageModel:
    supports_few_shot = True

    def __init__(self, model, processor, *, model_name, snapshot, mode, example_subject=None):
        self.model = model.eval()
        self.processor = processor
        self.mode = mode
        self.telemetry = []
        self.current = {}
        size = dict(processor.image_processor.size)
        if (size.get("longest_edge") != 16777216 or size.get("shortest_edge") != 65536
                or not processor.image_processor.do_resize):
            raise ValueError(f"Processor differs from the original experiment: {size}")
        self.inference_config = {
            "model_source": model_name, "model_revision": Path(snapshot).name,
            "device": str(model.device), "dtype": str(model.dtype),
            "attention": "efficient_sdpa_explicit_kv",
            "max_new_tokens": 128, "do_sample": False, "processor_size": size,
            "processor_do_resize": processor.image_processor.do_resize,
            "processor_config": processor.to_dict(), "image_views": "one_whole_image",
            "image_conversion": "RGB", "exif_transpose": False, "detail_crop_size": None,
            "custom_pixel_cap": None, "passes": 1,
        }
        def check_logits(_module, _arguments, output):
            import torch
            if not torch.isfinite(output).all():
                raise FloatingPointError("Non-finite raw Qwen logits")
        self.model.lm_head.register_forward_hook(check_logits)
        if mode == "few_shot":
            self.inference_config.update(reference_subject=example_subject, reference_images_per_target=4,
                total_images_per_conversation=5,
                reference_conversation="four_labeled_user_assistant_pairs_then_target")

    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen3.5-0.8B", *, cache_dir=None, revision=None,
                        device="cuda", mode="zero_shot", example_subject=None):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if mode not in {"zero_shot", "few_shot"}:
            raise ValueError("Qwen mode must be zero_shot or few_shot")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Connect to a GPU runtime or explicitly select CPU.")
        snapshot = snapshot_download(repo_id=model_name, revision=revision, cache_dir=cache_dir)
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        if device != "cuda" or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
        checks = validate_efficient_attention()
        model = AutoModelForMultimodalLM.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.float16,
            attn_implementation="sdpa", device_map={"": 0})
        if {str(p.device) for p in model.parameters()} != {"cuda:0"}:
            raise RuntimeError("Unexpected CPU offload or multi-GPU placement")
        adapter = cls(model, processor, model_name=model_name, snapshot=snapshot,
                      mode=mode, example_subject=example_subject)
        adapter.inference_config["attention_validation"] = checks
        return adapter

    def _generate(self, image_path, prompt, *, examples=()):
        import torch

        if (self.mode == "few_shot" and len(examples) != 4) or (self.mode == "zero_shot" and examples):
            raise ValueError("Few-shot mode requires four references; zero-shot mode requires none")
        with ExitStack() as stack:
            def image_turn(path, text):
                source = stack.enter_context(Image.open(path))
                image = stack.enter_context(source.convert("RGB"))
                return {"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": text}]}

            messages = []
            for index, example in enumerate(examples):
                reference_prompt = ((prompt + "\n\n") if index == 0 else "")
                messages.append(image_turn(example.image_path,
                    reference_prompt + "Reference image: classify its degradation."))
                messages.append({"role": "assistant", "content": [
                    {"type": "text", "text": json.dumps(example.prediction)}]})
            messages.append(image_turn(image_path,
                "Target image: return only its JSON classification." if examples else prompt))
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
            self.current["input_shapes"] = {key: list(value.shape) for key, value in inputs.items()
                                            if hasattr(value, "shape")}
            self.current["image_grid_thw"] = inputs["image_grid_thw"].tolist()
            with torch.inference_mode(), efficient_attention():
                outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            self.current["generated_tokens"] = int(generated.numel())
            return self.processor.decode(generated, skip_special_tokens=True).strip()

    def generate(self, image_path, prompt, *, examples=()):
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.current = {"image_path": str(image_path), "status": "running",
                        "allocated_before_mib": torch.cuda.memory_allocated() / 1024**2}
        started = perf_counter()
        try:
            result = self._generate(image_path, prompt, examples=examples)
            self.current["status"] = "generated"
            return result
        except Exception as error:
            self.current.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
                                error=f"{type(error).__name__}: {error}")
            raise
        finally:
            torch.cuda.synchronize()
            self.current.update(elapsed_seconds=perf_counter() - started,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                allocated_after_mib=torch.cuda.memory_allocated() / 1024**2)
            self.telemetry.append(dict(self.current))
            if self.current["status"] == "oom":
                torch.cuda.empty_cache()
