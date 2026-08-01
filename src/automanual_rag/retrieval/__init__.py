"""Retrieval backends."""

from .bm25 import BM25Index, build_bm25_index
from .dense import DenseIndex, build_dense_index
from .hybrid import HybridIndex
from .table import TableIndex, build_table_index
from .table_rows import TableRowIndex, build_table_row_index
from .visual import (
    VisualIndex,
    VisualTextFusionIndex,
    build_visual_index,
)
from automanual_rag.graphrag import GraphRetriever, build_manual_graph

__all__ = [
    "BM25Index",
    "DenseIndex",
    "HybridIndex",
    "TableIndex",
    "TableRowIndex",
    "VisualIndex",
    "VisualTextFusionIndex",
    "GraphRetriever",
    "build_bm25_index",
    "build_dense_index",
    "build_table_index",
    "build_table_row_index",
    "build_visual_index",
    "build_manual_graph",
]
