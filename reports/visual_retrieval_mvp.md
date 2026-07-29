# Visual retrieval MVP

This stage adds traceable image-to-image retrieval over the four Ford 2026
manuals. It does not add VLM answering, OCR, a web UI, table re-parsing, or a
neural image encoder.

## Resource accounting

All 3,803 MinerU JPG resources are referenced exactly once by a normalized
element and exist locally:

| Document | `image` elements | `table` crops | Total JPG |
|---|---:|---:|---:|
| Bronco | 899 | 143 | 1,042 |
| F-150 Lightning | 945 | 144 | 1,089 |
| Mustang Mach-E | 665 | 107 | 772 |
| Maverick | 778 | 122 | 900 |
| **Total** | **3,287** | **516** | **3,803** |

The 516-file difference is therefore fully explained by normalized `table`
elements. The MVP indexes only the 3,287 `image` elements. Table crops remain
traceable by stable element ID/page/asset path, but are not mixed into
icon/part/operation-diagram retrieval. There are no orphan files, duplicate
asset references, or invalid paths.

## Encoder decision

The machine had NumPy and Pillow, but no PyTorch, torchvision, Transformers,
ONNX Runtime, CLIP/SigLIP package, or cached model weights.

- [OpenAI CLIP ViT-B/32](https://github.com/openai/CLIP) requires
  PyTorch/torchvision; its
  [safetensors weight](https://huggingface.co/openai/clip-vit-base-patch32/blob/refs%2Fpr%2F3/model.safetensors)
  is about 605 MB. The official CLIP code is
  [MIT licensed](https://github.com/openai/CLIP/blob/main/LICENSE).
- [Google SigLIP Base](https://huggingface.co/google/siglip-base-patch16-224)
  is Apache-2.0 and its
  [safetensors weight](https://huggingface.co/google/siglip-base-patch16-224/blob/main/model.safetensors)
  is about 813 MB.

Adding either model plus a CPU PyTorch runtime would not be a lightweight
offline dependency for this MVP. No model or CUDA package was downloaded.

The implemented backend is
`traditional_multifeature_visual_v1` (`semantic_embedding=false`,
`neural_embedding=false`). Its 1,296-dimensional vector combines:

- background-trimmed, aspect-preserving low-resolution shape/intensity;
- gradient magnitude and spatial orientation histograms;
- RGB, HSV, and grayscale histograms;
- horizontal/vertical ink projections;
- low-frequency FFT magnitudes.

All blocks and final vectors are L2-normalized. This supports edited/cropped
manual screenshots, but is not a semantic visual encoder.

## Gold queries

The set contains 12 queries: one dev and two test queries per vehicle.
Features and weights were fixed before the first metric run; no tuning was
performed afterward.

| Query | Model | Category | Split | Perturbation |
|---|---|---|---|---|
| `bronco_icon_001` | Bronco | Instrument icon | dev | crop, contrast, resize, JPEG |
| `bronco_cluster_002` | Bronco | Instrument cluster | test | crop, contrast, resize, JPEG |
| `bronco_tailgate_003` | Bronco | Operation diagram | test | crop, rotate, brightness, JPEG |
| `f150_charge_port_004` | F-150 Lightning | Vehicle part | dev | crop, contrast, resize, JPEG |
| `f150_tailgate_005` | F-150 Lightning | Operation diagram | test | crop, blur, resize, JPEG |
| `f150_cluster_006` | F-150 Lightning | Instrument cluster | test | crop, rotate, brightness, JPEG |
| `mache_adapter_007` | Mustang Mach-E | Operation diagram | dev | crop, contrast, resize, JPEG |
| `mache_frunk_008` | Mustang Mach-E | Operation diagram | test | crop, rotate, brightness, JPEG |
| `mache_tire_kit_009` | Mustang Mach-E | Component diagram | test | crop, blur, resize, JPEG |
| `maverick_wiper_010` | Maverick | Control diagram | dev | crop, contrast, resize, JPEG |
| `maverick_hood_011` | Maverick | Operation diagram | test | crop, rotate, brightness, JPEG |
| `maverick_anchor_012` | Maverick | Component location | test | crop, blur, resize, JPEG |

Each record stores the query path, metadata filters, gold element/page/asset,
source asset, transformation parameters, and source/query SHA-256. All
source/query hashes differ. The 12 query files total 193,536 bytes.

This is a reproducible near-duplicate/edited-screenshot benchmark derived from
official manual evidence. It is useful for MVP validation but likely
overestimates performance on independent phone photos, heavily occluded
screenshots, or semantically similar images from outside the manuals.

## Results

Primary test metrics use the eight held-out test queries:

| Backend | Split | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Metadata violations |
|---|---|---:|---:|---:|---:|---:|
| Visual-only | test (8) | 0.8750 | 1.0000 | 1.0000 | 0.9375 | 0 |
| Visual + text-hint RRF | test (8) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| Visual-only | all (12) | 0.9167 | 1.0000 | 1.0000 | 0.9583 | 0 |
| Visual + text-hint RRF | all (12) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |

The visual-only miss at rank 1 was `f150_tailgate_005`: a visually similar
manual-tailgate illustration ranked first and the correct power-tailgate
removal diagram ranked second.

The fusion backend uses an additional user-supplied short text hint. It fuses
the visual rank with the existing BM25 rank of chunks on the same physical
page (`top-50`, `k=60`, equal weights). Its improvement must not be described
as upload-only visual performance.

## Integrity audit

The generated index is approximately 9.2 MiB and stores a `[3287, 1296]`
float32 matrix plus evidence metadata in a pickle-free NPZ. The audit passed:

- repeat loading produced identical metadata and vectors;
- zero vector/metadata alignment mismatches;
- zero invalid indexed paths, non-finite values, zero vectors, or
  normalization failures;
- zero missing Gold elements;
- zero repeated-query determinism failures;
- zero brand/model/year/region/language/manual-type violations;
- 120 fused results had zero source-rank, source-score, RRF-formula, or
  metadata violations.

Every search result returns `element_id`, `doc_id`, one-based `page_no`,
`section_path`, `asset_path`, and score. Fusion additionally retains
`visual_rank`, `visual_score`, `text_rank`, `text_score`, and `fusion_score`.

## Reproduce

Install only the lightweight dependencies if needed:

```powershell
python -m pip install -r .\requirements-visual.txt
```

Then regenerate Gold queries, rebuild the local index, evaluate both backends,
and run the audit:

```powershell
python .\scripts\run_visual_stage.py
```

Machine-readable outputs:

- `data/eval/visual_questions.jsonl`
- `outputs/metrics/visual_traditional_baseline.json`
- `outputs/metrics/visual_text_fusion.json`
- `outputs/metrics/visual_comparison.json`
- `outputs/metrics/visual_retrieval_audit.json`

The generated index remains ignored at
`data/indexes/visual_traditional.npz`.
