# Evidence-constrained answering baseline

This stage converts filtered BM25 retrieval into an auditable Evidence Pack
and a citation-bearing extractive response. It does not use an LLM or VLM for
generation.

## Behavior

`extractive_evidence_v1`:

- requires a document ID or both model and year before retrieval;
- keeps the same brand/model/year/region/language/manual-type hard filters;
- reranks the top 10 BM25 chunks using BM25 strength, query-term coverage,
  section coverage, and a small procedure-evidence bonus;
- returns at most three evidence chunks with chunk/element IDs, retrieval
  rank and score, rerank score, vehicle metadata, section path, and physical
  PDF page;
- preserves ordered steps and Warning/Caution text rather than inventing a
  paraphrase;
- refuses weak evidence and procedure questions that only match related terms
  without providing an operation.

The default refusal gates are BM25 score `12.0` and deterministic confidence
`0.55`. These are baseline heuristics, not calibrated probabilities.

## Results

Evaluation uses the existing 30-question project set: 26 answerable and four
no-answer cases.

| Metric | Result |
|---|---:|
| Answer/refusal decision accuracy | 0.9667 (29/30) |
| Answerable response rate | 0.9615 (25/26) |
| Gold citation recall | 0.8846 (23/26) |
| No-answer accuracy | 1.0000 (4/4) |
| Metadata filter violations | 0 |

The one conservative false refusal was `maverick_image_026`: the correct Gold
evidence was present in the Evidence Pack, but confidence `0.507281` stayed
below the `0.55` answer threshold.

The three citation misses were `f150_proc_010`, `f150_proc_011`, and
`mache_proc_017`. They were answered from related manual evidence, but the
three-item reranked Evidence Pack did not contain the exact Gold element.
This distinguishes citation membership from answer/refusal behavior.

All four no-answer questions were refused. In particular, the Bronco manual
mentions a “Diesel Exhaust Fluid system” inside an emissions-law passage, but
does not provide the requested refill procedure. The procedure-support guard
therefore rejects the keyword match instead of presenting it as an answer.

## Interpretation

These are descriptive development-set results, not an untouched held-out
test. Citation recall checks exact Gold element membership; it does not score
semantic answer quality. The extractive response is a safe local fallback and
the contract for a later configurable LLM/VLM generation adapter.

Reproduce:

```powershell
python .\scripts\evaluate_answering.py
```

Machine-readable output:

- `outputs/metrics/answering_baseline.json`
