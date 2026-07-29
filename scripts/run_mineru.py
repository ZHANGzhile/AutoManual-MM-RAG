#!/usr/bin/env python3
"""Batch-parse automotive manuals with the MinerU CLI.

The script intentionally keeps MinerU's native output directory structure so
that Markdown image links and JSON asset paths remain valid.

Examples:
    python scripts/run_mineru.py
    python scripts/run_mineru.py --backend hybrid-engine --effort high
    python scripts/run_mineru.py --api-url http://127.0.0.1:8000
    python scripts/run_mineru.py --manifest data/manifests/corpus.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "parsed"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "corpus.csv"

SUPPORTED_BACKENDS = (
    "pipeline",
    "vlm-engine",
    "hybrid-engine",
    "vlm-http-client",
    "hybrid-http-client",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MinerU for every PDF in a directory or corpus manifest."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing local PDF files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root directory for MinerU results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional CSV with doc_id and local_filename columns. If omitted, "
            "the script scans --input-dir recursively and uses each file stem "
            "as doc_id."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default="pipeline",
        help="MinerU parsing backend (default: pipeline).",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "txt", "ocr"),
        default="auto",
        help="Parsing method for pipeline/hybrid backends (default: auto).",
    )
    parser.add_argument(
        "--effort",
        choices=("medium", "high"),
        default="medium",
        help="Hybrid parsing effort (default: medium).",
    )
    parser.add_argument(
        "--api-url",
        help="Reuse an existing mineru-api service, for example http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--model-source",
        choices=("huggingface", "modelscope"),
        help="Override MINERU_MODEL_SOURCE for this run.",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        help=(
            "Only parse this document ID. Repeat the option to select multiple "
            "documents."
        ),
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Enable table recognition (slower; disabled by default).",
    )
    parser.add_argument(
        "--render-threads",
        type=int,
        default=4,
        help="PDF rendering worker count (default: 4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run documents that already have a _SUCCESS.json marker.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with remaining PDFs when one document fails.",
    )
    return parser.parse_args()


def safe_doc_id(value: str) -> str:
    """Reject doc IDs that could escape the configured output directory."""
    value = value.strip()
    if not value or value in {".", ".."}:
        raise ValueError("doc_id must not be empty")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe doc_id: {value!r}")
    return value


def documents_from_manifest(
    manifest_path: Path, input_dir: Path
) -> list[tuple[str, Path]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    documents: list[tuple[str, Path]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"doc_id", "local_filename"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, start=2):
            doc_id = safe_doc_id(row["doc_id"])
            filename = row["local_filename"].strip()
            if not filename:
                raise ValueError(
                    f"Empty local_filename at {manifest_path}:{line_number}"
                )

            pdf_path = Path(filename)
            if not pdf_path.is_absolute():
                pdf_path = input_dir / pdf_path
            documents.append((doc_id, pdf_path.resolve()))

    return documents


def documents_from_directory(input_dir: Path) -> list[tuple[str, Path]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pdf_paths = sorted(
        path.resolve()
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    return [(safe_doc_id(path.stem), path) for path in pdf_paths]


def validate_documents(documents: Iterable[tuple[str, Path]]) -> list[tuple[str, Path]]:
    checked: list[tuple[str, Path]] = []
    seen_ids: set[str] = set()

    for doc_id, pdf_path in documents:
        if doc_id in seen_ids:
            raise ValueError(f"Duplicate doc_id: {doc_id}")
        seen_ids.add(doc_id)

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found for {doc_id}: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input is not a PDF for {doc_id}: {pdf_path}")
        checked.append((doc_id, pdf_path))

    if not checked:
        raise ValueError("No PDF files found")
    return checked


def mineru_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (result.stdout or result.stderr).strip()
    return output or "unknown"


def build_command(
    executable: str,
    pdf_path: Path,
    doc_output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        executable,
        "-p",
        str(pdf_path),
        "-o",
        str(doc_output_dir),
        "-b",
        args.backend,
    ]

    if args.backend in {"pipeline", "hybrid-engine", "hybrid-http-client"}:
        command.extend(["-m", args.method])
        command.extend(["-f", "false", "-t", str(args.tables).lower()])
    if args.backend in {"hybrid-engine", "hybrid-http-client"}:
        command.extend(["--effort", args.effort])
    if args.api_url:
        command.extend(["--api-url", args.api_url])
    return command


def find_outputs(doc_output_dir: Path) -> dict[str, list[str]]:
    """Return project-relative paths for the main MinerU artifacts."""

    patterns = {
        "markdown": ("*.md",),
        "content_list": ("*_content_list.json",),
        "content_list_v2": ("*_content_list_v2.json",),
        "middle": ("*_middle.json",),
        "images": ("*.png", "*.jpg", "*.jpeg", "*.webp"),
        "tables": ("*.html",),
    }
    outputs: dict[str, list[str]] = {}
    for label, globs in patterns.items():
        matches = sorted(
            path.relative_to(doc_output_dir).as_posix()
            for pattern in globs
            for path in doc_output_dir.rglob(pattern)
            if path.is_file()
        )
        if matches:
            outputs[label] = matches
    return outputs


def write_success_marker(
    marker_path: Path,
    *,
    doc_id: str,
    pdf_path: Path,
    command: list[str],
    version: str,
    elapsed_seconds: float,
    outputs: dict[str, list[str]],
) -> None:
    payload = {
        "doc_id": doc_id,
        "source_pdf": str(pdf_path),
        "mineru_version": version,
        "command": command,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "outputs": outputs,
    }
    marker_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    executable = shutil.which("mineru")
    if executable is None:
        venv_executable = Path(sys.executable).with_name(
            "mineru.exe" if os.name == "nt" else "mineru"
        )
        if venv_executable.is_file():
            executable = str(venv_executable)
    if executable is None:
        print(
            "Error: MinerU CLI was not found. Install MinerU and confirm that "
            "`mineru --version` works in this terminal.",
            file=sys.stderr,
        )
        return 2

    manifest_path = args.manifest
    if manifest_path is None and DEFAULT_MANIFEST.is_file():
        manifest_path = DEFAULT_MANIFEST

    try:
        if manifest_path is not None:
            documents = documents_from_manifest(
                manifest_path.resolve(), args.input_dir.resolve()
            )
        else:
            documents = documents_from_directory(args.input_dir.resolve())
        documents = validate_documents(documents)
        if args.doc_id:
            requested_ids = set(args.doc_id)
            known_ids = {doc_id for doc_id, _ in documents}
            unknown_ids = sorted(requested_ids - known_ids)
            if unknown_ids:
                raise ValueError(
                    f"Unknown document ID(s): {', '.join(unknown_ids)}"
                )
            documents = [
                document for document in documents if document[0] in requested_ids
            ]
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    version = mineru_version(executable)
    env = os.environ.copy()
    env["MINERU_TABLE_ENABLE"] = str(args.tables).lower()
    env["MINERU_TABLE_MERGE_ENABLE"] = str(args.tables).lower()
    env["MINERU_FORMULA_ENABLE"] = "false"
    env["MINERU_PDF_RENDER_THREADS"] = str(max(1, args.render_threads))
    if args.model_source:
        env["MINERU_MODEL_SOURCE"] = args.model_source

    print(f"MinerU: {version}")
    print(f"Documents: {len(documents)}")

    failures: list[str] = []
    for index, (doc_id, pdf_path) in enumerate(documents, start=1):
        doc_output_dir = args.output_dir.resolve() / doc_id
        marker_path = doc_output_dir / "_SUCCESS.json"

        if marker_path.is_file() and not args.force:
            print(f"[{index}/{len(documents)}] Skip {doc_id}: already completed")
            continue

        doc_output_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(
            executable=executable,
            pdf_path=pdf_path,
            doc_output_dir=doc_output_dir,
            args=args,
        )

        print(f"[{index}/{len(documents)}] Parse {doc_id}")
        started_at = time.perf_counter()
        result = subprocess.run(command, check=False, env=env)
        elapsed_seconds = time.perf_counter() - started_at

        if result.returncode != 0:
            failures.append(doc_id)
            print(
                f"Failed: {doc_id} (exit={result.returncode}, "
                f"elapsed={elapsed_seconds:.1f}s)",
                file=sys.stderr,
            )
            if not args.keep_going:
                return result.returncode
            continue

        outputs = find_outputs(doc_output_dir)
        if not outputs.get("markdown") or not (
            outputs.get("content_list") or outputs.get("content_list_v2")
        ):
            failures.append(doc_id)
            print(
                f"Failed validation: no Markdown/content-list output for {doc_id}",
                file=sys.stderr,
            )
            if not args.keep_going:
                return 1
            continue

        write_success_marker(
            marker_path,
            doc_id=doc_id,
            pdf_path=pdf_path,
            command=command,
            version=version,
            elapsed_seconds=elapsed_seconds,
            outputs=outputs,
        )
        print(f"Completed: {doc_id} ({elapsed_seconds:.1f}s)")

    if failures:
        print(f"Failed documents: {', '.join(failures)}", file=sys.stderr)
        return 1

    print("All documents completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
