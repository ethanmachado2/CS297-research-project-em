import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import nbformat
from PIL import Image
import torch

from src.image_quality_pipeline import FewShotExample
from src.qwen_adapter import QwenImageModel
from src.trm_adapter import EXPERIMENT1_PROMPT

PROJECT = Path(__file__).resolve().parents[1]


class CloudNotebookTests(unittest.TestCase):
    def test_notebooks_are_cleared_valid_python_and_match_generator(self):
        spec = importlib.util.spec_from_file_location("builder", PROJECT / "scripts/build_colab_notebooks.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        for path in (PROJECT / "notebooks/colab_enterprise").glob("*.ipynb"):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            self.assertEqual(notebook, builder.build(path.stem))
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    self.assertIsNone(cell.execution_count)
                    self.assertEqual(cell.outputs, [])
                    ast.parse(cell.source)

    def test_cloud_prompt_matches_all_original_notebooks(self):
        prompt = (PROJECT / "configs/qwen_original_prompt.txt").read_text()
        self.assertEqual(prompt, EXPERIMENT1_PROMPT)
        for path in (PROJECT / "notebooks").glob("*before-resize-cap*.ipynb"):
            notebook = nbformat.read(path, as_version=4)
            assignments = [statement for cell in notebook.cells if cell.cell_type == "code"
                           for statement in ast.walk(ast.parse(cell.source))
                           if isinstance(statement, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "ORIGINAL_PROMPT"
                                   for t in statement.targets)]
            self.assertEqual(ast.literal_eval(assignments[0].value), prompt)


class QwenAdapterTests(unittest.TestCase):
    def test_conversations_preserve_zero_and_few_shot_contracts(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Processor:
            image_processor = SimpleNamespace(size={"longest_edge": 16777216, "shortest_edge": 65536}, do_resize=True)
            def to_dict(self):
                return {}
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.roles = [message["role"] for message in messages]
                self.texts = [part["text"] for message in messages for part in message["content"] if part["type"] == "text"]
                self.modes = [part["image"].mode for message in messages for part in message["content"] if part["type"] == "image"]
                return Inputs(input_ids=torch.tensor([[1, 2]]))
            def decode(self, tokens, **kwargs):
                self.tokens = tokens.tolist()
                return '{"noise":"0","severity":"0","route":"0"}'

        class Model:
            device = "cpu"
            dtype = torch.float32
            def eval(self):
                return self
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.tensor([[1, 2, 8, 9]])

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.new("L", (16, 16)).save(path)
            for mode in ("zero_shot", "few_shot"):
                processor, model = Processor(), Model()
                adapter = QwenImageModel(model, processor, model_name="test", snapshot="snapshots/sha", mode=mode)
                examples = [FewShotExample(str(path), "0", {"noise": "0", "severity": str(i), "route": "0"})
                            for i in range(4)] if mode == "few_shot" else []
                adapter.generate(path, "original prompt", examples=examples)
                self.assertEqual(processor.roles, ["user", "assistant"] * 4 + ["user"] if examples else ["user"])
                self.assertEqual(processor.tokens, [8, 9])
                self.assertEqual(processor.modes, ["RGB"] * (5 if examples else 1))
                self.assertTrue(processor.texts[0].startswith("original prompt"))
                self.assertEqual(model.kwargs["max_new_tokens"], 128)
                self.assertFalse(model.kwargs["do_sample"])
                if examples:
                    with self.assertRaises(ValueError):
                        adapter.generate(path, "prompt")


if __name__ == "__main__":
    unittest.main()
