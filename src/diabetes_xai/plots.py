"""Publication-quality figures generated from real model outputs only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

PALETTE = [
    "#2a6f97",
    "#e07a5f",
    "#3d9970",
    "#9c6ade",
    "#f2a541",
    "#5c677d",
    "#c44569",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
    "#bcbd22",
    "#1f77b4",
]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.fontsize": 8,
        "legend.frameon": False,
    }
)


PNG_DPI = 180  # web/README version; the PDF is vector (use it for papers)


def save(fig, path: Path) -> list[Path]:
    """Write ``path.pdf`` (vector, for publication) and a compact ``path.png`` (for the web)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = path.with_suffix(".png"), path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=PNG_DPI)
    plt.close(fig)
    _compact_png(png)
    return [png, pdf]


def _compact_png(p: Path) -> None:
    """Palette-quantise the PNG (about 3x smaller, visually identical for plots)."""
    try:
        from PIL import Image

        with Image.open(p) as im:
            q = im.convert("RGB").quantize(
                colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
            )
        q.save(p, optimize=True)
    except Exception:  # pragma: no cover - Pillow is a matplotlib dependency, but stay safe
        pass


def roc_curves(y, probas: dict[str, np.ndarray], title: str, path: Path, top: int | None = None):
    from sklearn.metrics import roc_auc_score

    items = sorted(probas.items(), key=lambda kv: -roc_auc_score(y, kv[1]))
    if top:
        items = items[:top]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for i, (name, p) in enumerate(items):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(
            fpr, tpr, lw=1.6, color=PALETTE[i % len(PALETTE)], label=f"{name} (AUC {roc_auc_score(y, p):.3f})"
        )
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=title,
        xlim=(-0.01, 1.01),
        ylim=(-0.01, 1.01),
    )
    ax.legend(loc="lower right")
    return save(fig, path)


def pr_curves(y, probas: dict[str, np.ndarray], title: str, path: Path, top: int | None = None):
    from sklearn.metrics import average_precision_score

    items = sorted(probas.items(), key=lambda kv: -average_precision_score(y, kv[1]))
    if top:
        items = items[:top]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for i, (name, p) in enumerate(items):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(
            rec,
            prec,
            lw=1.6,
            color=PALETTE[i % len(PALETTE)],
            label=f"{name} (AP {average_precision_score(y, p):.3f})",
        )
    ax.axhline(np.mean(y), ls=":", color="grey", lw=1, label=f"Prevalence {np.mean(y):.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="upper right")
    return save(fig, path)


def calibration(y, probas: dict[str, np.ndarray], title: str, path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for i, (name, p) in enumerate(probas.items()):
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=3, lw=1.4, color=PALETTE[i % len(PALETTE)], label=name)
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1, label="Perfect calibration")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left")
    return save(fig, path)


def confusion_grid(y, probas: dict[str, np.ndarray], thresholds: dict[str, float], title: str, path: Path):
    n = len(probas)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.1 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (name, p) in zip(axes.ravel(), probas.items(), strict=False):
        t = thresholds.get(name, 0.5)
        cm = confusion_matrix(y, (np.asarray(p) >= t).astype(int), labels=[0, 1])
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(
                c,
                r,
                f"{v:,}\n({pct[r, c]:.0%})",
                ha="center",
                va="center",
                color="white" if pct[r, c] > 0.6 else "black",
                fontsize=8,
            )
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(f"{name}\n(threshold {t:.2f})", fontsize=9)
        ax.grid(False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return save(fig, path)


def metric_bars(
    table: pd.DataFrame, metric: str, ci_lo: str | None, ci_hi: str | None, title: str, path: Path
):
    t = table.sort_values(metric)
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(t) + 1.2))
    err = None
    if ci_lo and ci_hi and ci_lo in t and ci_hi in t:
        err = np.vstack([t[metric] - t[ci_lo], t[ci_hi] - t[metric]])
    ax.barh(t["model"], t[metric], xerr=err, color="#2a6f97", alpha=0.85, capsize=2, error_kw={"lw": 0.8})
    for i, v in enumerate(t[metric]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    lo = max(0, t[metric].min() - 0.08)
    ax.set_xlim(lo, min(1, t[metric].max() + 0.06))
    ax.set(xlabel=metric.replace("_", " ").upper() if len(metric) < 8 else metric, title=title)
    ax.grid(axis="y", alpha=0)
    return save(fig, path)


def feature_ranking(table: pd.DataFrame, k: int, title: str, path: Path):
    rank_cols = [c for c in table.columns if c.endswith("Rank") and c != "Consensus Rank"]
    t = table.set_index("Feature")[rank_cols + ["Consensus Rank"]]
    fig, ax = plt.subplots(figsize=(0.75 * len(t.columns) + 3, 0.32 * len(t) + 1.5))
    im = ax.imshow(t.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(t.columns)), [c.replace(" Rank", "") for c in t.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(t)), t.index)
    for (r, c), v in np.ndenumerate(t.values):
        ax.text(
            c,
            r,
            f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if v > len(t) * 0.55 else "black",
        )
    ax.axhline(k - 0.5, color="red", lw=1.5, ls="--")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, label="Rank (1 = most important)")
    return save(fig, path)


def protocol_comparison(df: pd.DataFrame, metric: str, title: str, path: Path):
    """Grouped bars: leaky vs leakage-safe protocol (mean +/- sd across CV folds)."""
    models = df["model"].unique().tolist()
    protocols = df["protocol"].unique().tolist()
    x = np.arange(len(models))
    w = 0.8 / len(protocols)
    fig, ax = plt.subplots(figsize=(1.0 * len(models) + 2.5, 4))
    colors = {
        "leakage-safe": "#2a6f97",
        "leaky: class-median imputation": "#f2a541",
        "leaky (notebook protocol)": "#c44569",
    }
    for i, prot in enumerate(protocols):
        sub = df[df["protocol"] == prot].set_index("model").reindex(models)
        ax.bar(
            x + i * w - 0.4 + w / 2,
            sub[f"{metric}_mean"],
            w,
            yerr=sub[f"{metric}_std"],
            label=prot,
            color=colors.get(prot, PALETTE[i]),
            capsize=2,
        )
    ax.set_xticks(x, models, rotation=25, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set(ylabel=metric.replace("_", " "), title=title)
    ax.legend()
    return save(fig, path)


def shap_summary(shap_values, X: pd.DataFrame, title: str, path: Path, max_display: int = 15):  # noqa: N803
    import shap

    shap.summary_plot(shap_values, X, show=False, max_display=max_display, plot_size=(7, 5.5))
    fig = plt.gcf()
    fig.axes[0].set_title(title, fontweight="bold")
    return save(fig, path)


def shap_bar(mean_abs: pd.Series, title: str, path: Path):
    s = mean_abs.sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 0.3 * len(s) + 1.2))
    ax.barh(s.index, s.values, color="#3d9970")
    ax.set(xlabel="mean(|SHAP value|)  (impact on log-odds)", title=title)
    ax.grid(axis="y", alpha=0)
    return save(fig, path)
