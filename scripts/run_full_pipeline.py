#!/usr/bin/env python3
"""Run the complete public-PDF-to-evaluated-demo pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
REPORT = PROJECT_ROOT / "outputs" / "metrics" / "full_pipeline_run.json"
STAGES = (
    "download",
    "mineru",
    "import",
    "chunks",
    "indexes",
    "evaluate",
    "test",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the complete AutoManual-MM-RAG project.",
    )
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume the official PDFs were already integrity-verified.",
    )
    parser.add_argument(
        "--skip-mineru",
        action="store_true",
        help="Use existing validated MinerU outputs.",
    )
    parser.add_argument(
        "--mineru-python",
        type=Path,
        help="Python executable from the isolated MinerU environment.",
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Enable MinerU table recognition for a slower full reparse.",
    )
    parser.add_argument(
        "--force-mineru",
        action="store_true",
        help="Reparse documents even when _SUCCESS.json exists.",
    )
    return parser.parse_args()


def selected_stages(start: str, stop: str) -> list[str]:
    start_index = STAGES.index(start)
    stop_index = STAGES.index(stop)
    if start_index > stop_index:
        raise ValueError("--from-stage must not come after --to-stage")
    return list(STAGES[start_index : stop_index + 1])


def default_mineru_python() -> Path:
    if sys.platform == "win32":
        return PROJECT_ROOT / ".venv-mineru" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv-mineru" / "bin" / "python"


def script_command(
    python: Path | str,
    script_name: str,
    *arguments: str,
) -> list[str]:
    return [
        str(python),
        str(SCRIPTS / script_name),
        *arguments,
    ]


def stage_commands(
    stage: str,
    *,
    project_python: Path | str,
    mineru_python: Path,
    tables: bool,
    force_mineru: bool,
) -> list[list[str]]:
    if stage == "download":
        return [
            script_command(
                project_python,
                "download_public_manuals.py",
            )
        ]
    if stage == "mineru":
        command = script_command(
            mineru_python,
            "run_mineru.py",
            "--manifest",
            str(PROJECT_ROOT / "data" / "manifests" / "corpus.csv"),
            "--keep-going",
        )
        if tables:
            command.append("--tables")
        if force_mineru:
            command.append("--force")
        return [command]
    if stage == "import":
        return [
            script_command(
                project_python,
                "import_mineru_output.py",
                "--strict",
            )
        ]
    if stage == "chunks":
        return [script_command(project_python, "build_chunks.py")]
    if stage == "indexes":
        return [
            script_command(project_python, name)
            for name in (
                "build_bm25_index.py",
                "build_dense_index.py",
                "build_table_index.py",
                "build_table_row_index.py",
                "build_visual_index.py",
                "build_graph_index.py",
            )
        ]
    if stage == "evaluate":
        return [
            script_command(project_python, name)
            for name in (
                "evaluate_bm25.py",
                "evaluate_retrieval.py",
                "build_visual_eval.py",
                "evaluate_visual.py",
                "evaluate_tables.py",
                "evaluate_table_rows.py",
                "evaluate_answering.py",
                "evaluate_agentic_graphrag.py",
                "audit_retrieval.py",
                "audit_visual.py",
            )
        ]
    if stage == "test":
        return [
            [
                str(project_python),
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ]
        ]
    raise ValueError(f"Unknown stage: {stage}")


def portable_command(command: Sequence[str]) -> list[str]:
    portable: list[str] = []
    for part in command:
        try:
            portable.append(
                Path(part).resolve().relative_to(PROJECT_ROOT).as_posix()
            )
        except (OSError, ValueError):
            portable.append(part)
    return portable


def write_report(report: dict[str, object]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(REPORT)


def main() -> int:
    args = parse_args()
    try:
        stages = selected_stages(args.from_stage, args.to_stage)
        if args.skip_download and "download" in stages:
            stages.remove("download")
        if args.skip_mineru and "mineru" in stages:
            stages.remove("mineru")
        mineru_python = (
            args.mineru_python.resolve()
            if args.mineru_python
            else default_mineru_python().resolve()
        )
        if "mineru" in stages and not mineru_python.is_file():
            raise FileNotFoundError(
                "MinerU Python not found. Run "
                "powershell -ExecutionPolicy Bypass "
                "-File scripts/setup_mineru.ps1 first."
            )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    report: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "requested_range": {
            "from": args.from_stage,
            "to": args.to_stage,
        },
        "skipped": {
            "download": args.skip_download,
            "mineru": args.skip_mineru,
        },
        "stages": [],
    }
    write_report(report)
    pipeline_started = time.perf_counter()
    for stage in stages:
        commands = stage_commands(
            stage,
            project_python=sys.executable,
            mineru_python=mineru_python,
            tables=args.tables,
            force_mineru=args.force_mineru,
        )
        stage_result = {
            "stage": stage,
            "status": "running",
            "commands": [portable_command(command) for command in commands],
        }
        report["stages"].append(stage_result)
        write_report(report)
        stage_started = time.perf_counter()
        print(f"\n=== {stage.upper()} ===", flush=True)
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
            )
            if completed.returncode:
                stage_result["status"] = "failed"
                stage_result["returncode"] = completed.returncode
                stage_result["elapsed_seconds"] = round(
                    time.perf_counter() - stage_started,
                    3,
                )
                report["status"] = "failed"
                report["finished_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                write_report(report)
                return completed.returncode
        stage_result["status"] = "completed"
        stage_result["elapsed_seconds"] = round(
            time.perf_counter() - stage_started,
            3,
        )
        write_report(report)

    report["status"] = "completed"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = round(
        time.perf_counter() - pipeline_started,
        3,
    )
    write_report(report)
    print(f"\nPipeline completed. Report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
