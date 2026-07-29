# MinerU output quality spot check

Date: 2026-07-29

Corpus: four Ford 2026 North American English owner manuals
Parser: MinerU 3.4.4, `pipeline` backend, `txt` method, formula and dedicated
table recognition disabled

## Result

The native outputs are suitable as evidence-preserving input for the next
retrieval stage. The strict normalized import produced 33,934 elements with no
missing element page numbers, broken asset paths, metadata leakage, or importer
anomalies.

`page_no` is the one-based physical PDF page (`page_idx + 1`). The page number
printed inside a manual is separately retained as
`source_locator.source_page_label`.

| Document | PDF pages | Native elements | Text | Image | Table | Missing page | Invalid asset | Anomaly |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ford_bronco_2026_na_en` | 578 | 10,644 | 7,752 | 899 | 143 | 0 | 0 | 0 |
| `ford_f150_lightning_2026_na_en` | 636 | 12,170 | 9,013 | 945 | 144 | 0 | 0 | 0 |
| `ford_mache_2026_na_en` | 468 | 8,573 | 6,270 | 665 | 107 | 0 | 0 | 0 |
| `ford_maverick_2026_na_en` | 534 | 9,752 | 7,096 | 778 | 122 | 0 | 0 | 0 |
| **Total** | **2,216** | **41,139** | **30,131** | **3,287** | **516** | **0** | **0** | **0** |

The normalized image count includes MinerU `image`, `chart`, and `equation`
assets. Together with the 516 table crops, all 3,803 extracted asset files are
accounted for and exist locally.

## Representative visual checks

The checks compared rendered source PDF pages with native `content_list`,
`content_list_v2`, and `middle` elements, then verified the normalized JSONL
page, section, bbox, asset path, and neighbor text. Reading order is reported
for the natural two-column flow.

| Document | PDF page | Printed page | Coverage | Result |
|---|---:|---:|---|---|
| Bronco | 85 | 81 | Tailgate diagrams, two step sequences, Warning and Notes | Pass: four image crops exist; left column then right column order and adjacent text align. |
| Bronco | 262 | 258 | Cruise-control Warning, two steps, icon, information table | Pass with table limit: warning/steps/icon align; table is a valid crop with empty structured HTML. |
| Bronco | 294 | 290 | 11-step roof-rack operation, two diagrams, Warning, capacity table | Pass with table limit: step and image order align; caption is retained, cells are image-only. |
| F-150 Lightning | 44 | 40 | Child-restraint diagrams and two compatibility tables | Pass with table limit: both diagrams and both table crops align; table cells are not structured. |
| F-150 Lightning | 47 | 43 | Child-seat steps 7-10, diagrams, Warning and Note | Pass: text order, warning block, and diagram neighbors align. |
| F-150 Lightning | 219 | 215 | Charging diagram/icons, Notes, AC charging Caution | Pass: the rare `CAUTION:` block is preserved and associated with the charging section. |
| Mustang Mach-E | 37 | 33 | Child-restraint diagram and two compatibility tables | Pass with table limit: page and assets align; table cells are image-only. |
| Mustang Mach-E | 95 | 91 | Steering-wheel Warning/Notes, three steps, diagram and icons | Pass: page header anchors `Steering Wheel`; sequence and visual neighbors align. |
| Mustang Mach-E | 175 | 171 | Charging diagram/icons, Notes, AC charging Caution and step start | Pass: Caution and first charging step remain in order; assets resolve. |
| Maverick | 81 | 77 | Two-column tailgate procedures, two diagrams, Warnings and Notes | Pass: page header fixes the section to `Tailgate`; left/right sequences and images align. |
| Maverick | 89 | 85 | Washer-fluid diagram, three steps, capacity/material tables | Pass with table limit: procedure order and asset links align; both tables are image-only. |
| Maverick | 348 | 344 | Hood Warning/steps, diagram, indicator icon and two tables | Pass with table limit: two-column order and icon explanation align; tables have no cell HTML. |

The native content lists contain no elements on PDF pages 2, 4, and the
penultimate page of each manual (12 pages total). All 12 were rendered and
visually confirmed to be blank, so these are not missed content pages.

## Structural checks

- All 20 JSON files (four MinerU JSON files per document plus four
  `_SUCCESS.json` markers) deserialize. This corrects an earlier inventory that
  counted 19.
- All 41,139 native content-list elements have valid JSON structure; layout
  headers, footers, page labels, and empty text boxes are excluded from
  retrieval elements but remain locatable in the native source.
- Running page headers provide the first section component; MinerU title levels
  provide nested components. This prevents cross-page section leakage.
- Warning, Caution, and Note text is not rewritten or discarded. The source
  scan found 1,651 Warning blocks, two Caution blocks, and 3,141 Note blocks.
- Image/table content combines section, native caption/footnote, available
  recognized text, and nearest previous/next same-page text. No VLM captioning
  was used.
- Each element records its native content-list path, source index/type,
  zero-based source page index, one-based page, printed page label, bbox, and
  native asset path. `source_span` is `null` because MinerU did not provide
  character offsets.
- An identical strict re-import produced byte-identical per-document JSONL,
  confirming stable rebuilds and stable element IDs.
- Tests validate schema round trips, page conversion, deterministic/cross-doc
  IDs, four-document metadata isolation, asset existence, adjacency links, and
  malformed-field handling.

## Known limitations and follow-up boundary

1. Dedicated table recognition was disabled. MinerU still detected 516 table
   regions and saved valid table images, but every table has empty HTML. The
   current `content` is caption/section/neighbor evidence, not a trustworthy
   cell transcription. Numeric table QA must wait for selective table-enabled
   re-parsing of necessary pages.
2. Minor source text defects remain, mainly joined title words such as
   `INSTALLINGTHE` or `HIGHBEAM`, missing spaces around dash separators, and an
   occasional word-level recognition typo. Evidence paths and rendered table or
   image crops allow later correction without losing provenance.
3. Image captions are often empty in the native output. The importer supplies
   reliable local assets and neighboring born-digital text, but does not
   invent visual descriptions or call an expensive VLM.
4. BM25, dense, table-row, and visual vector indexes are intentionally outside
   this stage.
