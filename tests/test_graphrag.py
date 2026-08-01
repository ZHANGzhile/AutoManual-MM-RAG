import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from automanual_rag.graphrag import (
    EDGE_TYPES,
    NODE_TYPES,
    GraphRetriever,
    build_manual_graph,
)


class GraphRAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifests" / "corpus.csv"
        self.processed = self.root / "processed"
        self.index = self.root / "indexes" / "graph.sqlite3"
        self.manifest.parent.mkdir(parents=True)
        rows = [
            {
                "doc_id": "ford_alpha_2026_na_en",
                "brand": "Ford",
                "model": "Alpha",
                "year": "2026",
                "region": "North America",
                "language": "en",
                "manual_type": "owner_manual",
                "source_url": "https://example.com/alpha.pdf",
                "downloaded_at": "2026-01-01T00:00:00+00:00",
                "local_filename": "alpha.pdf",
            },
            {
                "doc_id": "ford_beta_2026_na_en",
                "brand": "Ford",
                "model": "Beta",
                "year": "2026",
                "region": "North America",
                "language": "en",
                "manual_type": "owner_manual",
                "source_url": "https://example.com/beta.pdf",
                "downloaded_at": "2026-01-01T00:00:00+00:00",
                "local_filename": "beta.pdf",
            },
        ]
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            self._write_document(row)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _base(row, page_no, section_path):
        return {
            "doc_id": row["doc_id"],
            "brand": row["brand"],
            "model": row["model"],
            "year": row["year"],
            "region": row["region"],
            "language": row["language"],
            "manual_type": row["manual_type"],
            "page_no": page_no,
            "section_path": section_path,
        }

    def _write_document(self, row) -> None:
        directory = self.processed / row["doc_id"]
        directory.mkdir(parents=True)
        section = ["Steering Wheel", "ADJUSTING THE STEERING WHEEL"]
        chunks = [
            {
                **self._base(row, 10, section),
                "chunk_id": f"{row['doc_id']}:steps",
                "chunk_type": "steps",
                "content": (
                    "1. Unlock the steering column.\n"
                    "2. Adjust the steering wheel.\n"
                    "3. Lock the steering column."
                ),
                "element_ids": [f"{row['doc_id']}:text:steps"],
                "page_start": 10,
            },
            {
                **self._base(row, 10, section),
                "chunk_id": f"{row['doc_id']}:warning",
                "chunk_type": "warning",
                "content": "WARNING: Do not adjust while driving.",
                "element_ids": [f"{row['doc_id']}:text:warning"],
                "page_start": 10,
            },
            {
                **self._base(row, 11, ["Specifications", "Wheel Torque"]),
                "chunk_id": f"{row['doc_id']}:spec",
                "chunk_type": "text",
                "content": "Wheel torque specification is shown in the table.",
                "element_ids": [f"{row['doc_id']}:text:spec"],
                "page_start": 11,
            },
            {
                **self._base(row, 12, ["Symbols Glossary", "Brake Indicator"]),
                "chunk_id": f"{row['doc_id']}:symbol",
                "chunk_type": "text",
                "content": "The brake warning symbol indicator means check brakes.",
                "element_ids": [f"{row['doc_id']}:text:symbol"],
                "page_start": 12,
            },
            {
                **self._base(row, 13, ["Battery", "Battery Caution"]),
                "chunk_id": f"{row['doc_id']}:caution",
                "chunk_type": "caution",
                "content": "CAUTION: Use the specified battery.",
                "element_ids": [f"{row['doc_id']}:text:caution"],
                "page_start": 13,
            },
        ]
        elements = [
            {
                **self._base(row, 12, ["Symbols Glossary", "Brake Indicator"]),
                "element_id": f"{row['doc_id']}:image",
                "element_type": "image",
                "content": "Brake warning symbol",
                "asset_path": f"data/parsed/{row['doc_id']}/symbol.jpg",
                "source_locator": {"source_type": "image"},
            },
            {
                **self._base(row, 11, ["Specifications", "Wheel Torque"]),
                "element_id": f"{row['doc_id']}:table",
                "element_type": "table",
                "content": "Wheel torque specification table",
                "asset_path": f"data/parsed/{row['doc_id']}/table.jpg",
                "source_locator": {"source_type": "table"},
            },
        ]
        for name, values in (("chunks.jsonl", chunks), ("elements.jsonl", elements)):
            (directory / name).write_text(
                "\n".join(json.dumps(value) for value in values) + "\n",
                encoding="utf-8",
            )

    def test_build_contains_required_types_and_provenance(self) -> None:
        summary = build_manual_graph(
            manifest_path=self.manifest,
            processed_root=self.processed,
            output_path=self.index,
        )
        self.assertEqual(summary["documents"], 2)
        self.assertTrue(NODE_TYPES.issubset(summary["node_counts"]))
        self.assertTrue(EDGE_TYPES.issubset(summary["edge_counts"]))
        self.assertTrue(all(summary["node_counts"][name] > 0 for name in NODE_TYPES))
        self.assertTrue(all(summary["edge_counts"][name] > 0 for name in EDGE_TYPES))
        connection = sqlite3.connect(self.index)
        try:
            missing = connection.execute(
                """
                SELECT COUNT(*) FROM edges
                WHERE doc_id = '' OR evidence_ids = '[]'
                """
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(missing, 0)

    def test_retrieval_expands_paths_without_cross_vehicle_nodes(self) -> None:
        build_manual_graph(
            manifest_path=self.manifest,
            processed_root=self.processed,
            output_path=self.index,
        )
        retriever = GraphRetriever(self.index)
        results = retriever.search(
            "adjust steering wheel steps and warning",
            filters={"model": "Alpha", "year": "2026"},
            limit=5,
        )
        self.assertTrue(results)
        self.assertTrue(all(item["doc_id"] == "ford_alpha_2026_na_en" for item in results))
        self.assertTrue(all(1 <= item["hops"] <= 2 for item in results))
        self.assertIn("HAS_WARNING", {relation for item in results for relation in item["relations"]})
        for path in results:
            self.assertTrue(
                all(node["doc_id"] == "ford_alpha_2026_na_en" for node in path["nodes"])
            )

    def test_retrieval_requires_exact_vehicle_context(self) -> None:
        build_manual_graph(
            manifest_path=self.manifest,
            processed_root=self.processed,
            output_path=self.index,
        )
        with self.assertRaises(ValueError):
            GraphRetriever(self.index).search(
                "steering wheel",
                filters={"brand": "Ford"},
            )


if __name__ == "__main__":
    unittest.main()
