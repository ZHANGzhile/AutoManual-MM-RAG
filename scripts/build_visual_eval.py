#!/usr/bin/env python3
"""Create reproducible perturbed visual queries from official evidence crops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.ingestion.mineru import load_manifest
from automanual_rag.schema import ManualElement


GOLD_DEFINITIONS = (
    {
        "query_id": "bronco_icon_001",
        "element_id": "ford_bronco_2026_na_en:p0025:image:000296:78b2c71fdf6be67b",
        "category": "instrument_icon",
        "split": "dev",
        "query_text": "fasten seatbelt warning symbol",
        "transform": "icon_crop_jpeg",
    },
    {
        "query_id": "bronco_cluster_002",
        "element_id": "ford_bronco_2026_na_en:p0112:image:002180:9ee1e65d0958db40",
        "category": "instrument_cluster",
        "split": "test",
        "query_text": "Bronco instrument cluster overview",
        "transform": "crop_contrast_jpeg",
    },
    {
        "query_id": "bronco_tailgate_003",
        "element_id": "ford_bronco_2026_na_en:p0084:image:001530:0c8972f4c6969366",
        "category": "operation_diagram",
        "split": "test",
        "query_text": "tailgate emergency locking key blade",
        "transform": "rotate_brightness_jpeg",
    },
    {
        "query_id": "f150_charge_port_004",
        "element_id": "ford_f150_lightning_2026_na_en:p0218:image:004326:6dacbaf791a4d6bf",
        "category": "vehicle_part",
        "split": "dev",
        "query_text": "locating the charge port door",
        "transform": "crop_contrast_jpeg",
    },
    {
        "query_id": "f150_tailgate_005",
        "element_id": "ford_f150_lightning_2026_na_en:p0102:image:001856:50f3030173ef59fa",
        "category": "operation_diagram",
        "split": "test",
        "query_text": "removing and installing the power tailgate",
        "transform": "blur_resize_jpeg",
    },
    {
        "query_id": "f150_cluster_006",
        "element_id": "ford_f150_lightning_2026_na_en:p0141:image:002694:6691867fb545a033",
        "category": "instrument_cluster",
        "split": "test",
        "query_text": "F-150 Lightning instrument cluster power gauge",
        "transform": "rotate_brightness_jpeg",
    },
    {
        "query_id": "mache_adapter_007",
        "element_id": "ford_mache_2026_na_en:p0168:image:003293:d6522f38390fc6e3",
        "category": "operation_diagram",
        "split": "dev",
        "query_text": "connect the NACS adapter to the vehicle",
        "transform": "crop_contrast_jpeg",
    },
    {
        "query_id": "mache_frunk_008",
        "element_id": "ford_mache_2026_na_en:p0310:image:006019:a56b7c536f32a35f",
        "category": "operation_diagram",
        "split": "test",
        "query_text": "open front luggage compartment with no vehicle power",
        "transform": "rotate_brightness_jpeg",
    },
    {
        "query_id": "mache_tire_kit_009",
        "element_id": "ford_mache_2026_na_en:p0364:image:006842:69e1b0dbf36dc75f",
        "category": "component_diagram",
        "split": "test",
        "query_text": "tire sealant and inflator kit components",
        "transform": "blur_resize_jpeg",
    },
    {
        "query_id": "maverick_wiper_010",
        "element_id": "ford_maverick_2026_na_en:p0086:image:001596:bd68f5212122846d",
        "category": "control_diagram",
        "split": "dev",
        "query_text": "switching windshield wipers on and off",
        "transform": "crop_contrast_jpeg",
    },
    {
        "query_id": "maverick_hood_011",
        "element_id": "ford_maverick_2026_na_en:p0347:image:006593:fe3a9f5454b6b501",
        "category": "operation_diagram",
        "split": "test",
        "query_text": "opening the hood release latch",
        "transform": "rotate_brightness_jpeg",
    },
    {
        "query_id": "maverick_anchor_012",
        "element_id": "ford_maverick_2026_na_en:p0038:image:000638:cfda2a444144a48a",
        "category": "component_location",
        "split": "test",
        "query_text": "child restraint lower anchor point locations",
        "transform": "blur_resize_jpeg",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual Gold queries.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "corpus.csv",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--query-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "visual_queries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "visual_questions.jsonl",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_images(
    manifest_path: Path,
    processed_root: Path,
) -> dict[str, ManualElement]:
    images: dict[str, ManualElement] = {}
    for document in load_manifest(manifest_path):
        path = processed_root / document.doc_id / "elements.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    element = ManualElement.from_dict(json.loads(line))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if element.element_type == "image":
                    images[element.element_id] = element
    return images


def _crop_fraction(image: Any, amount: float) -> Any:
    left = max(0, round(image.width * amount))
    top = max(0, round(image.height * amount * 0.75))
    right = min(image.width, round(image.width * (1.0 - amount * 0.8)))
    bottom = min(image.height, round(image.height * (1.0 - amount)))
    if right - left < 8 or bottom - top < 8:
        return image
    return image.crop((left, top, right, bottom))


def _transform(source: Path, name: str) -> tuple[Any, dict[str, Any]]:
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError as exc:
        raise RuntimeError("Visual Gold generation requires Pillow") from exc
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    source_size = list(image.size)

    if name == "icon_crop_jpeg":
        image = _crop_fraction(image, 0.025)
        image = ImageEnhance.Contrast(image).enhance(1.12)
        image = image.resize(
            (round(image.width * 1.35), round(image.height * 1.35)),
            resampling,
        )
        details = {
            "crop_fraction": 0.025,
            "contrast": 1.12,
            "resize_scale": 1.35,
            "jpeg_quality": 78,
        }
    elif name == "crop_contrast_jpeg":
        image = _crop_fraction(image, 0.04)
        image = ImageEnhance.Contrast(image).enhance(1.08)
        image = image.resize(
            (round(image.width * 0.82), round(image.height * 0.82)),
            resampling,
        )
        details = {
            "crop_fraction": 0.04,
            "contrast": 1.08,
            "resize_scale": 0.82,
            "jpeg_quality": 82,
        }
    elif name == "rotate_brightness_jpeg":
        image = _crop_fraction(image, 0.03)
        image = image.rotate(
            1.2,
            resample=getattr(Image, "Resampling", Image).BICUBIC,
            expand=False,
            fillcolor="white",
        )
        image = ImageEnhance.Brightness(image).enhance(0.94)
        image = image.resize(
            (round(image.width * 0.88), round(image.height * 0.88)),
            resampling,
        )
        details = {
            "crop_fraction": 0.03,
            "rotation_degrees": 1.2,
            "brightness": 0.94,
            "resize_scale": 0.88,
            "jpeg_quality": 84,
        }
    elif name == "blur_resize_jpeg":
        image = _crop_fraction(image, 0.02)
        image = image.filter(ImageFilter.GaussianBlur(radius=0.55))
        image = ImageEnhance.Brightness(image).enhance(1.04)
        image = image.resize(
            (round(image.width * 0.76), round(image.height * 0.76)),
            resampling,
        )
        details = {
            "crop_fraction": 0.02,
            "gaussian_blur_radius": 0.55,
            "brightness": 1.04,
            "resize_scale": 0.76,
            "jpeg_quality": 76,
        }
    else:
        raise ValueError(f"Unknown visual query transform: {name}")
    return image, {
        "name": name,
        "source_size": source_size,
        "query_size": list(image.size),
        **details,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in records
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    try:
        images = _load_images(
            args.manifest.resolve(),
            args.processed_root.resolve(),
        )
        query_root = args.query_root.resolve()
        query_root.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for definition in GOLD_DEFINITIONS:
            element = images.get(definition["element_id"])
            if element is None:
                raise ValueError(
                    f"Gold element not found: {definition['element_id']}"
                )
            if element.page_no is None or not element.asset_path:
                raise ValueError(
                    f"Gold element is not traceable: {element.element_id}"
                )
            source = (PROJECT_ROOT / element.asset_path).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Gold source image missing: {source}")
            query_path = query_root / f"{definition['query_id']}.jpg"
            image, transform = _transform(source, definition["transform"])
            temporary = query_path.with_suffix(".tmp")
            image.save(
                temporary,
                format="JPEG",
                quality=int(transform["jpeg_quality"]),
                optimize=True,
            )
            temporary.replace(query_path)
            source_hash = _sha256(source)
            query_hash = _sha256(query_path)
            if source_hash == query_hash:
                raise ValueError(
                    f"Query is byte-identical to source: {query_path}"
                )
            records.append(
                {
                    "query_id": definition["query_id"],
                    "category": definition["category"],
                    "split": definition["split"],
                    "query_image": query_path.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    "query_text": definition["query_text"],
                    "filters": {
                        "brand": element.brand,
                        "model": element.model,
                        "year": element.year,
                        "region": element.region,
                        "language": element.language,
                        "manual_type": element.manual_type,
                    },
                    "gold_evidence": {
                        "element_id": element.element_id,
                        "doc_id": element.doc_id,
                        "page_no": element.page_no,
                        "section_path": list(element.section_path),
                        "asset_path": element.asset_path,
                    },
                    "source": {
                        "source_asset_path": element.asset_path,
                        "source_sha256": source_hash,
                        "query_sha256": query_hash,
                        "derived_from_official_manual_evidence": True,
                    },
                    "transform": transform,
                }
            )
        _write_jsonl(args.output.resolve(), records)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Built {len(records)} visual queries: "
        f"{sum(r['split'] == 'dev' for r in records)} dev, "
        f"{sum(r['split'] == 'test' for r in records)} test"
    )
    print(f"Manifest: {args.output.resolve()}")
    print(f"Images: {args.query_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
