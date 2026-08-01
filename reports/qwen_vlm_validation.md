# Qwen VLM live validation

## Scope

This report records the first paid external VLM smoke test for the project.
It validates provider authentication and the complete image-grounded RAG path;
it is not presented as a full generated-answer quality benchmark.

No API key, workspace ID, account identifier, or full endpoint hostname is
stored in this repository.

## Environment

| Field | Value |
|---|---|
| Provider | Alibaba Cloud Model Studio |
| Region | Germany (Frankfurt) |
| Deployment scope | EU |
| Model requested | `qwen3-vl-flash` |
| Model returned | `qwen3-vl-flash` |
| Protocol | OpenAI-compatible Chat Completions |
| Thinking mode | Disabled |
| Maximum generated tokens | 700 |

The local provider adapter reads `DASHSCOPE_API_KEY`, `QWEN_MODEL`, and
`DASHSCOPE_CHAT_COMPLETIONS_URL` from the ignored `.env` file.

## Authentication smoke

A minimal text-only request returned:

| Check | Result |
|---|---|
| HTTP status | 200 |
| Returned model | `qwen3-vl-flash` |
| Response text | `OK` |

This establishes that the API key, selected model, workspace URL, and
Frankfurt region are mutually compatible.

## Multimodal end-to-end smoke

| Field | Value |
|---|---|
| Query ID | `bronco_icon_001` |
| Query asset | `data/eval/visual_queries/bronco_icon_001.jpg` |
| Question | `What does this warning symbol mean?` |
| Required metadata | Ford / Bronco / 2026 / North America / English |
| Retrieval backend | `visual_text_rrf` |
| Retrieved evidence items | 5 |
| Cross-vehicle evidence | 0 |
| Generation backend | `qwen_chat_completions_v1` |
| Final status | `answered` |
| Generation status | `generated` |

The model identified the uploaded symbol as the seatbelt reminder indicator.
Its answer cited Evidence `[5]`, the Bronco 2026 Symbols Glossary safety entry
on physical PDF page 29, which points to the seatbelt reminder indicators
section.

The response passed the application's structural grounding check: it was
non-empty, used at least one citation, and every citation ID existed in the
five-item Evidence Pack.

## Cost and limitations

The authentication request and one multimodal request together were estimated
to cost less than USD 0.002 at the published sub-32K Frankfurt EU rates. The
API response usage fields were not persisted, so this is an estimate rather
than an invoice value.

Limitations:

- This is one manually inspected generated answer.
- The existing 12-query visual benchmark measures retrieval, not generated
  answer faithfulness.
- The returned answer included more explanation than necessary; a production
  evaluation should score concise evidence faithfulness in addition to the
  current citation-ID validation.
- Network or provider errors deliberately fall back to the offline extractive
  result.

## Reproduction

After placing a Frankfurt API key and workspace-specific Chat Completions URL
in the ignored `.env` file:

```powershell
.\.venv\Scripts\python.exe .\scripts\answer_image_question_qwen.py `
  .\data\eval\visual_queries\bronco_icon_001.jpg `
  --question "What does this warning symbol mean?" `
  --brand Ford `
  --model Bronco `
  --year 2026 `
  --json
```

The real `.env` must never be committed.
