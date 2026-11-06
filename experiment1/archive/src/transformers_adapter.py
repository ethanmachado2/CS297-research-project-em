"""Batched Transformers vision adapter using an already loaded model/processor."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageOps

from .image_quality_pipeline import FewShotExample


class TransformersImageModel:
    """Run decoder-only vision models (including the notebook's Qwen model).

    Images are independent conversations. Left padding aligns generated tokens
    across variable-length inputs; decoding excludes the full padded prompt.
    Optional detail crops add a native center view to each reference/target.
    Qwen pixel budgets must include the crop area to avoid resizing that view.
    No model downloads or loading take place here.
    """

    supports_few_shot = True

    def __init__(self, model: Any, processor: Any, *, max_new_tokens: int = 128,
                 detail_crop_size: int | None = None, full_resolution: bool = False):
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        if detail_crop_size is not None and (
            isinstance(detail_crop_size, bool) or not isinstance(detail_crop_size, int)
            or detail_crop_size < 256 or detail_crop_size % 32
        ):
            raise ValueError("detail_crop_size must be a multiple of 32, at least 256, or None")
        self.detail_crop_size = detail_crop_size
        if not isinstance(full_resolution, bool):
            raise ValueError("full_resolution must be a boolean")
        if full_resolution and detail_crop_size is not None:
            raise ValueError("full_resolution and detail_crop_size are mutually exclusive")
        self.full_resolution = full_resolution
        image_processor = getattr(processor, "image_processor", None)
        self.patch_multiple = (getattr(image_processor, "patch_size", 0)
                               * getattr(image_processor, "merge_size", 0))
        if full_resolution and not self.patch_multiple:
            raise ValueError("full_resolution requires processor patch_size and merge_size")
        size = getattr(getattr(processor, "image_processor", None), "size", {})
        if detail_crop_size and "longest_edge" in size and "shortest_edge" in size:
            pixels = detail_crop_size ** 2
            if not size["shortest_edge"] <= pixels <= size["longest_edge"]:
                raise ValueError("Processor pixel budget would resize the native detail crop")
        self.processor.tokenizer.padding_side = "left"
        self.model.eval()

    @property
    def inference_config(self) -> dict[str, Any]:
        """JSON metadata used to prevent comparing differently configured runs."""
        image_processor = getattr(self.processor, "image_processor", None)
        size = getattr(image_processor, "size", None)
        model_config = getattr(self.model, "config", None)
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "dtype": str(getattr(self.model, "dtype", "unknown")),
            "device": str(getattr(self.model, "device", "unknown")),
            "model_source": getattr(model_config, "_name_or_path", None),
            "processor_size": dict(size) if size is not None else None,
            "detail_crop_size": self.detail_crop_size,
            "image_views": ("full_resolution" if self.full_resolution else
                            "overview_and_center_detail" if self.detail_crop_size else "overview"),
            "full_resolution": self.full_resolution,
            "padding": {"mode": "edge", "placement": "right_bottom", "multiple": self.patch_multiple}
                       if self.full_resolution else None,
            "processor_do_resize": False if self.full_resolution else getattr(image_processor, "do_resize", True),
            "exif_transpose": True,
            "model_revision": getattr(model_config, "_commit_hash", None),
        }

    def generate(self, image_path: str | Path, prompt: str, *, examples: Sequence[FewShotExample] = ()) -> str:
        return self.generate_batch([image_path], prompt, examples=examples)[0]

    def generate_batch(
        self, image_paths: Sequence[str | Path], prompt: str, *, examples: Sequence[FewShotExample] = (),
    ) -> list[str]:
        import torch

        if not image_paths:
            return []
        with ExitStack() as stack:
            def image_turn(path, text):
                source = stack.enter_context(Image.open(path))
                oriented = stack.enter_context(ImageOps.exif_transpose(source))
                image = stack.enter_context(oriented.convert("RGB"))
                if self.full_resolution:
                    import numpy as np

                    width, height = image.size
                    right, bottom = (-width) % self.patch_multiple, (-height) % self.patch_multiple
                    if right or bottom:
                        # Edge padding preserves every source pixel without interpolation.
                        padded = np.pad(np.asarray(image), ((0, bottom), (0, right), (0, 0)), mode="edge")
                        image = stack.enter_context(Image.fromarray(padded))
                content = []
                if self.detail_crop_size:
                    width, height = image.size
                    edge = self.detail_crop_size
                    if min(width, height) < edge:
                        raise ValueError(f"Image {path} is too small for a native {edge}x{edge} detail crop")
                    left, top = (width - edge) // 2, (height - edge) // 2
                    detail = stack.enter_context(image.crop((left, top, left + edge, top + edge)))
                    content.extend([
                        {"type": "text", "text": "Overview of this image:"},
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Native-resolution center detail of the SAME image:"},
                        {"type": "image", "image": detail},
                    ])
                else:
                    content.append({"type": "image", "image": image})
                content.append({"type": "text", "text": text})
                return {"role": "user", "content": content}

            references = []
            for index, example in enumerate(examples):
                text = (prompt + "\n\n" if index == 0 else "") + "Reference image: classify its degradation."
                references.append(image_turn(example.image_path, text))
                references.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(example.prediction)}],
                })
            conversations = []
            for path in image_paths:
                target_prompt = "Target image: return only its JSON classification." if examples else prompt
                conversations.append(references + [image_turn(path, target_prompt)])
            processor_kwargs = {"padding": True}
            if self.full_resolution:
                processor_kwargs["images_kwargs"] = {"do_resize": False}
            inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                processor_kwargs=processor_kwargs,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            if self.full_resolution:
                expected = [item["image"].size for turns in conversations for turn in turns
                            for item in turn["content"] if item["type"] == "image"]
                patch_size = self.processor.image_processor.patch_size
                actual = [(int(grid[2]) * patch_size, int(grid[1]) * patch_size)
                          for grid in inputs["image_grid_thw"]]
                if actual != expected:
                    raise ValueError(f"Processor changed full-resolution dimensions: {expected} -> {actual}")
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            generated = outputs[:, inputs["input_ids"].shape[-1]:]
            return [text.strip() for text in self.processor.batch_decode(generated, skip_special_tokens=True)]


__all__ = ["TransformersImageModel"]
