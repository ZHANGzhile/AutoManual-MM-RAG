from pathlib import Path
import unittest

from scripts.run_full_pipeline import (
    PROJECT_ROOT,
    selected_stages,
    stage_commands,
)


class FullPipelineTests(unittest.TestCase):
    def test_stage_range_is_ordered_and_inclusive(self) -> None:
        self.assertEqual(
            selected_stages("import", "indexes"),
            ["import", "chunks", "indexes"],
        )
        with self.assertRaises(ValueError):
            selected_stages("test", "download")

    def test_index_stage_builds_every_runtime_index(self) -> None:
        commands = stage_commands(
            "indexes",
            project_python="python",
            mineru_python=Path("mineru-python"),
            tables=False,
            force_mineru=False,
        )
        scripts = [Path(command[1]).name for command in commands]
        self.assertEqual(
            scripts,
            [
                "build_bm25_index.py",
                "build_dense_index.py",
                "build_table_index.py",
                "build_table_row_index.py",
                "build_visual_index.py",
                "build_graph_index.py",
            ],
        )

    def test_mineru_stage_uses_isolated_python_and_flags(self) -> None:
        mineru_python = PROJECT_ROOT / ".venv-mineru" / "python.exe"
        command = stage_commands(
            "mineru",
            project_python="python",
            mineru_python=mineru_python,
            tables=True,
            force_mineru=True,
        )[0]
        self.assertEqual(command[0], str(mineru_python))
        self.assertIn("--tables", command)
        self.assertIn("--force", command)
        self.assertIn("--keep-going", command)

    def test_evaluation_rebuilds_visual_gold_before_scoring(self) -> None:
        commands = stage_commands(
            "evaluate",
            project_python="python",
            mineru_python=Path("mineru-python"),
            tables=False,
            force_mineru=False,
        )
        scripts = [Path(command[1]).name for command in commands]
        self.assertLess(
            scripts.index("build_visual_eval.py"),
            scripts.index("evaluate_visual.py"),
        )


if __name__ == "__main__":
    unittest.main()
