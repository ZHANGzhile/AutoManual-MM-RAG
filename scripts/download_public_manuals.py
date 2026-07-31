#!/usr/bin/env python3
"""Download the official public manuals with an integrity lock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "corpus.csv"
DEFAULT_CHECKSUMS = (
    PROJECT_ROOT / "data" / "manifests" / "manual_checksums.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw"
BUFFER_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official manuals listed in corpus.csv.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--doc-id",
        action="append",
        help="Download only this document ID; repeat to select more.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a valid existing PDF.",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Allow a download without the locked SHA-256 (not reproducible).",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(BUFFER_SIZE):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(value: str) -> str:
    value = value.strip()
    if (
        not value
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or not value.casefold().endswith(".pdf")
    ):
        raise ValueError(f"Unsafe PDF filename: {value!r}")
    return value


def load_downloads(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    downloads: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"doc_id", "source_url", "local_filename"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Manifest is missing required fields: "
                + ", ".join(sorted(missing))
            )
        for line_number, row in enumerate(reader, start=2):
            doc_id = row["doc_id"].strip()
            url = row["source_url"].strip()
            filename = safe_filename(row["local_filename"])
            if not doc_id:
                raise ValueError(f"Empty doc_id at {path}:{line_number}")
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(
                    f"Official source must use HTTPS at {path}:{line_number}"
                )
            downloads.append(
                {
                    "doc_id": doc_id,
                    "source_url": url,
                    "local_filename": filename,
                }
            )
    if not downloads:
        raise ValueError("Manifest contains no downloads")
    return downloads


def load_checksums(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Checksum lock not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Checksum lock must be a JSON object")
    return value


def verify_pdf(
    path: Path,
    expected: dict[str, Any] | None,
    *,
    require_checksum: bool,
) -> tuple[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    size = path.stat().st_size
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError(f"Downloaded file is not a PDF: {path}")
    digest = sha256_file(path)
    if require_checksum and expected is None:
        raise ValueError(f"No checksum lock entry for {path.name}")
    if expected is not None:
        expected_size = int(expected["size_bytes"])
        expected_hash = str(expected["sha256"]).casefold()
        if size != expected_size:
            raise ValueError(
                f"Size mismatch for {path.name}: {size} != {expected_size}"
            )
        if digest != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: {digest}"
            )
    return size, digest


def download_one(
    url: str,
    destination: Path,
    timeout: float,
    expected: dict[str, Any] | None,
    *,
    require_checksum: bool,
) -> tuple[int, str]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    request = Request(
        url,
        headers={"User-Agent": "AutoManual-MM-RAG/1.0"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Download returned HTTP {response.status}: {url}"
                )
            with temporary.open("wb") as output:
                while block := response.read(BUFFER_SIZE):
                    output.write(block)
        size, digest = verify_pdf(
            temporary,
            expected,
            require_checksum=require_checksum,
        )
        temporary.replace(destination)
        return size, digest
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    try:
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        downloads = load_downloads(args.manifest.resolve())
        checksums = load_checksums(args.checksums.resolve())
        if args.doc_id:
            requested = set(args.doc_id)
            known = {item["doc_id"] for item in downloads}
            unknown = sorted(requested - known)
            if unknown:
                raise ValueError(
                    "Unknown doc_id(s): " + ", ".join(unknown)
                )
            downloads = [
                item for item in downloads if item["doc_id"] in requested
            ]
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(downloads, start=1):
            destination = output_dir / item["local_filename"]
            expected = checksums.get(item["local_filename"])
            if destination.is_file() and not args.force:
                size, digest = verify_pdf(
                    destination,
                    expected,
                    require_checksum=not args.skip_checksum,
                )
                print(
                    f"[{index}/{len(downloads)}] Verified existing "
                    f"{item['doc_id']} ({size} bytes, {digest[:12]}...)"
                )
                continue
            print(
                f"[{index}/{len(downloads)}] Download {item['doc_id']} "
                f"from {item['source_url']}"
            )
            size, digest = download_one(
                item["source_url"],
                destination,
                args.timeout,
                expected,
                require_checksum=not args.skip_checksum,
            )
            print(
                f"Verified {destination.name} "
                f"({size} bytes, {digest[:12]}...)"
            )
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Official manuals ready: {len(downloads)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
