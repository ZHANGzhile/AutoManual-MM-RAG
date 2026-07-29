# MinerU local run

The four Ford manuals in this project are born-digital PDFs, so the fastest
local route is MinerU's `pipeline` backend with the `txt` method. Formula and
table recognition are disabled by default in the batch script; images, text,
Markdown, and structured JSON are still extracted.

## Recreate the temporary environment

Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv-mineru
.\.venv-mineru\Scripts\python.exe -m pip install "mineru[all]"
New-Item -ItemType Directory -Force .\.cache\mineru\modelscope | Out-Null
$env:MODELSCOPE_CACHE = (Resolve-Path .\.cache\mineru\modelscope).Path
$env:MINERU_TOOLS_CONFIG_JSON = (Join-Path (Get-Location) ".cache\mineru\mineru.json")
$env:MINERU_MODEL_SOURCE = "modelscope"
.\.venv-mineru\Scripts\mineru-models-download.exe -s modelscope -m pipeline
```

ModelScope was used because the Hugging Face large-file endpoint did not
respond reliably on this machine.

## Parse manuals

Parse every PDF under `data/raw`:

```powershell
.\.venv-mineru\Scripts\python.exe .\scripts\run_mineru.py `
  --method txt `
  --model-source modelscope `
  --keep-going
```

Parse only selected manuals by repeating `--doc-id`:

```powershell
.\.venv-mineru\Scripts\python.exe .\scripts\run_mineru.py `
  --method txt `
  --model-source modelscope `
  --doc-id ford_bronco_2026_na_en `
  --doc-id ford_maverick_2026_na_en
```

Add `--tables` only when table recognition is required; it downloads more
models and takes longer.

## Outputs

Each manual is written below `data/parsed/<doc_id>/`. MinerU keeps its native
subdirectory structure so Markdown image references remain valid. A successful
batch-script run also writes `_SUCCESS.json`.

## Remove MinerU after parsing

Once all outputs have been verified, close any terminal using the environment,
then remove only these temporary directories:

```powershell
Remove-Item -LiteralPath .\.venv-mineru -Recurse -Force
Remove-Item -LiteralPath .\.cache\mineru -Recurse -Force
```

The source PDFs, generated Markdown, JSON, and images are not removed.
