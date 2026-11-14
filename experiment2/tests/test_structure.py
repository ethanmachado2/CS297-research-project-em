"""Source/notebook checks requiring only Python's standard library."""
import ast
import json
from pathlib import Path
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class StructureTests(unittest.TestCase):
    def test_python_sources_compile_and_have_no_sibling_or_cloud_imports(self):
        for path in sorted(PROJECT.rglob("*.py")):
            if any(part.startswith(".") or part in {"outputs", "data", "__pycache__"}
                   for part in path.relative_to(PROJECT).parts):
                continue
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
            compile(tree, str(path), "exec")
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [name.name for name in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                self.assertFalse(any(name.startswith(("google", "experiment1")) for name in modules), path)

    def test_three_clean_notebooks_compile_and_call_matching_workload(self):
        notebooks = sorted((PROJECT / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 3)
        for path in notebooks:
            notebook = json.loads(path.read_text())
            self.assertEqual(notebook["nbformat"], 4)
            cells = notebook["cells"]
            self.assertEqual(len({cell["id"] for cell in cells}), len(cells))
            code = []
            for index, cell in enumerate(cells):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                compile(source, f"{path.name}:cell{index}", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                code.append(source)
            source = "\n".join(code)
            self.assertIn(f'"src.run", "{path.stem}"', source)
            self.assertIn('"--gpu", str(GPU)', source)
            self.assertIn('SELECTION = "full"', source)
            self.assertNotIn("jupyter-proxy", source)

    def test_defaults_are_fp16_and_original_prompt_is_present(self):
        trm = json.loads((PROJECT / "configs/trm.json").read_text())
        qwen = json.loads((PROJECT / "configs/qwen.json").read_text())
        self.assertEqual(trm["training"]["precision"], "float16")
        self.assertEqual(trm["training"]["device"], "cuda")
        self.assertEqual(trm["split"]["counts"], [35, 5, 10])
        self.assertRegex(qwen["revision"], "^[a-f0-9]{40}$")
        self.assertTrue((PROJECT / "configs/qwen_prompt.txt").read_text().strip())


if __name__ == "__main__":
    unittest.main()
