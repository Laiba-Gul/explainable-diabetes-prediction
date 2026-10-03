"""Write tables as CSV (machine-readable), Markdown (README) and LaTeX (papers)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    path_without_suffix.with_suffix(".md").write_text(to_markdown(df))
    try:
        latex = df.to_latex(index=False, float_format="%.4f", escape=True)
    except Exception:  # noqa: BLE001 - jinja2 may be missing; LaTeX is optional
        latex = ""
    if latex:
        path_without_suffix.with_suffix(".tex").write_text(latex)


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
