#!/usr/bin/env python3
"""Run the image-grounded RAG answer path with Qwen3-VL-Flash."""

from __future__ import annotations

import os

from answer_image_question import main


if __name__ == "__main__":
    os.environ["AUTOMANUAL_GENERATION_BACKEND"] = "qwen"
    raise SystemExit(main())
