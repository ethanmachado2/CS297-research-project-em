"""Reusable original-configuration Qwen inference for cloud notebooks."""
from contextlib import ExitStack
import json
from pathlib import Path

from PIL import Image


class QwenImageModel:
    supports_few_shot = True

    def __init__(self, model, processor, *, model_name, snapshot, mode, example_subject=None):
        self.model = model.eval()
        self.processor = processor
        self.mode = mode
        size = dict(processor.image_processor.size)
        if (size.get("longest_edge") != 16777216 or size.get("shortest_edge") != 65536
                or not processor.image_processor.do_resize):
            raise ValueError(f"Processor differs from the original experiment: {size}")
        self.inference_config = {
            "model_source": model_name, "model_revision": Path(snapshot).name,
            "device": str(model.device), "dtype": str(model.dtype),
            "max_new_tokens": 128, "do_sample": False, "processor_size": size,
            "processor_do_resize": processor.image_processor.do_resize,
            "processor_config": processor.to_dict(), "image_views": "one_whole_image",
            "image_conversion": "RGB", "exif_transpose": False, "detail_crop_size": None,
            "custom_pixel_cap": None, "passes": 1,
        }
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
        # Preserve the original experiment's float32 baseline; precision changes
        # should be separate experiment configurations.
        model = AutoModelForMultimodalLM.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.float32).to(device)
        return cls(model, processor, model_name=model_name, snapshot=snapshot,
                   mode=mode, example_subject=example_subject)

    def generate(self, image_path, prompt, *, examples=()):
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
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            return self.processor.decode(generated, skip_special_tokens=True).strip()
