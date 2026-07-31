# Reproducible demo cases

These cases use the local indexes built from the four Ford 2026 North
American English owner manuals. Start the app with:

```powershell
python .\scripts\launch_demo.py
```

## 1. Cited text procedure

- Vehicle: `Ford Bronco 2026`
- Tab: `Text question`
- Question: `How do I adjust the steering wheel?`
- Expected: three ordered steps, the movement Warning, a safety note, and
  physical PDF page 89 in the Evidence Pack.
- Screenshot: [`demo-text-answer.png`](../screenshots/demo-text-answer.png)

## 2. Image-to-manual retrieval

- Vehicle: `Ford Bronco 2026`
- Tab: `Image search`
- Image: `data/eval/visual_queries/bronco_cluster_002.jpg`
- Optional text hint: `Bronco instrument cluster overview`
- Expected: physical PDF page 112 at rank 1 under the selected vehicle hard
  filter.
- Screenshot: [`demo-image-search.png`](../screenshots/demo-image-search.png)

## 3. Verified exact table value

- Vehicle: `Ford Bronco 2026`
- Tab: `Table search`
- Question:
  `What is the maximum recommended roof load when the Bronco is in motion?`
- Expected: `110 lb (50 kg)`, physical PDF page 294, source image SHA-256,
  and the verified roof-load table crop shown first.
- Screenshot: [`demo-table-answer.png`](../screenshots/demo-table-answer.png)

Safety regression: asking for the hybrid Maverick fuel-tank capacity is
refused because the curated source explicitly excludes the full hybrid
variant.
