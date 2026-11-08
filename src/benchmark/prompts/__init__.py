"""Exact UTF-8 benchmark text; callers own substitutions and request shapes."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prompt_path(name: str) -> Path:
    return ROOT / name


def load_text(name: str) -> str:
    # Decode bytes directly: preserve whitespace, final newlines and CRLF.
    return prompt_path(name).read_bytes().decode("utf-8")
