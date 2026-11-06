import ast
import unittest

import nbformat

from trm_fixtures import PROJECT


class TRMNotebookTests(unittest.TestCase):
    def test_valid_cleared_notebook_and_python_cells(self):
        notebook = nbformat.read(PROJECT / "notebooks/cs297-experiment1-em-local-trm.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
                ast.parse(cell.source)
                self.assertNotIn("from transformers", cell.source)
        self.assertIn("export_saved_run", notebook.cells[-1].source)
