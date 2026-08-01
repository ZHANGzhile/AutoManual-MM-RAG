# Agentic GraphRAG evaluation

This report compares the unchanged BM25 answer baseline, standalone deterministic GraphRAG, and the explicit-state Agentic GraphRAG workflow on the checked-in multi-hop development set.

## Results

| System | Evidence recall | Path accuracy | Citation faithfulness | Route accuracy | Decision accuracy | Refusal accuracy | Metadata violations | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline RAG | 0.9167 | N/A | 1.0000 | N/A | 1.0000 | 1.0000 | 0 | 17.6 ms | 27.8 ms |
| GraphRAG | 0.9167 | 0.6250 | 1.0000 | N/A | 0.6667 | 0.0000 | 0 | 154.7 ms | 504.4 ms |
| Agentic GraphRAG | 0.9167 | 0.5714 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 135.5 ms | 490.1 ms |

No remote model was called in this comparison, so measured token usage is unavailable and API cost is USD 0.00.

## Failure cases

- `GraphRAG` / `mh_f150_charge_003`: {"decision": true, "path": false, "route": null}
- `Agentic GraphRAG` / `mh_f150_charge_003`: {"decision": true, "path": false, "route": true}
- `GraphRAG` / `mh_f150_child_restraint_004`: {"decision": true, "path": false, "route": null}
- `Agentic GraphRAG` / `mh_f150_child_restraint_004`: {"decision": true, "path": false, "route": true}
- `GraphRAG` / `mh_maverick_remote_battery_008`: {"decision": true, "path": false, "route": null}
- `Agentic GraphRAG` / `mh_maverick_remote_battery_008`: {"decision": true, "path": false, "route": true}
- `GraphRAG` / `mh_noanswer_f150_009`: {"decision": false, "path": null, "route": null}
- `GraphRAG` / `mh_noanswer_mache_010`: {"decision": false, "path": null, "route": null}
- `GraphRAG` / `mh_noanswer_maverick_011`: {"decision": false, "path": null, "route": null}
- `GraphRAG` / `mh_noanswer_bronco_012`: {"decision": false, "path": null, "route": null}

## Interpretation boundaries

- The questions are derived from the existing development corpus.
- Graph construction and final answers are deterministic and offline.
- Path accuracy requires one returned path to contain every Gold node type, relation, and page.
- Latency uses independent per-system graph caches; the first query for each vehicle is a cold load.
- Cost is zero only because this run does not call a remote generator.
