from pathlib import Path
import tempfile
import unittest

from automanual_rag.serialization import relativize_project_paths


class RelativizeProjectPathsTests(unittest.TestCase):
    def test_nested_project_paths_become_portable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            artifact = root / "data" / "indexes" / "dense.npz"
            value = {
                "index": artifact.as_posix(),
                "nested": [str(root), {"external": "https://example.com/a"}],
            }

            result = relativize_project_paths(value, root)

            self.assertEqual(result["index"], "data/indexes/dense.npz")
            self.assertEqual(result["nested"][0], ".")
            self.assertEqual(
                result["nested"][1]["external"],
                "https://example.com/a",
            )


if __name__ == "__main__":
    unittest.main()
