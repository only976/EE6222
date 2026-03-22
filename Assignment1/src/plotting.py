from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy_curve(
    df: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    vlines: list[tuple[float, str]] | None = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=220)
    ax.plot(df["dim"], df["accuracy"], marker="o", linewidth=1.5)

    if vlines:
        ymax = float(df["accuracy"].max()) if len(df) else 1.0
        for x, label in vlines:
            ax.axvline(x=x, linestyle="--", linewidth=1.2, color="#777777", alpha=0.9)
            ax.annotate(
                str(label),
                xy=(x, ymax),
                xytext=(3, -3),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8,
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )
    ax.set_xlabel("Dimension d")
    ax.set_ylabel("Classification accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_overlaid_curves(
    curves: list[tuple[str, pd.DataFrame]],
    *,
    title: str,
    out_path: Path,
    baseline: tuple[str, float] | None = None,
    vlines: list[tuple[float, str]] | None = None,
):
    """Overlay multiple (dim, accuracy) curves on one plot.

    Parameters
    ----------
    curves:
        List of (label, df) where df has columns ['dim', 'accuracy'].
    baseline:
        Optional (label, accuracy) to draw as a horizontal reference line.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=220)

    # Determine x-range from provided curves
    all_dims = []
    for _, df in curves:
        if "dim" in df.columns:
            all_dims.extend(df["dim"].to_list())
    xmin = min(all_dims) if all_dims else 0
    xmax = max(all_dims) if all_dims else 1

    if baseline is not None:
        base_label, base_acc = baseline
        ax.hlines(
            y=base_acc,
            xmin=xmin,
            xmax=xmax,
            colors="#555555",
            linestyles="--",
            linewidth=1.2,
            label=base_label,
        )

    if vlines:
        for x, label in vlines:
            ax.axvline(x=x, linestyle="--", linewidth=1.2, color="#777777", alpha=0.7)
            ax.text(
                x,
                0.02,
                str(label),
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
            )

    # Draw each reducer curve using different gray levels.
    gray_levels = ["#111111", "#333333", "#555555", "#777777", "#999999", "#BBBBBB"]
    for i, (label, df) in enumerate(curves):
        color = gray_levels[i % len(gray_levels)]
        ax.plot(df["dim"], df["accuracy"], marker="o", linewidth=1.3, color=color, label=label)

    ax.set_xlabel("Dimension d")
    ax.set_ylabel("Classification accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_overlaid_curves_highlight_one(
    curves: list[tuple[str, pd.DataFrame]],
    *,
    title: str,
    out_path: Path,
    highlight_label: str,
    baseline: tuple[str, float] | None = None,
    method_colors: dict[str, str] | None = None,
    vlines: list[tuple[float, str]] | None = None,
):
    """Overlay multiple curves; highlight only one curve and gray out others.

    Intended for 5x5 icon grids: for a fixed classifier (row), each column uses
    the same multi-curve plot but highlights a different reducer curve.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=240)

    # Determine x-range from provided curves
    all_dims: list[int] = []
    for _, df in curves:
        if "dim" in df.columns:
            all_dims.extend([int(x) for x in df["dim"].to_list()])
    xmin = min(all_dims) if all_dims else 0
    xmax = max(all_dims) if all_dims else 1

    def _color_for(label: str, *, default: str) -> str:
        if method_colors is None:
            return default
        return method_colors.get(label, default)

    # Baseline as a horizontal line; highlight it when requested.
    if baseline is not None:
        base_label, base_acc = baseline
        is_base_highlight = highlight_label == base_label
        ax.hlines(
            y=base_acc,
            xmin=xmin,
            xmax=xmax,
            colors=_color_for(base_label, default="#111111") if is_base_highlight else "#C7C7C7",
            linestyles="-" if is_base_highlight else "--",
            linewidth=2.4 if is_base_highlight else 1.2,
            alpha=1.0 if is_base_highlight else 0.9,
            label=base_label,
        )

        if is_base_highlight:
            # Annotate baseline accuracy
            ax.annotate(
                f"acc={base_acc:.3f}",
                xy=(xmax, base_acc),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=8,
                color=_color_for(base_label, default="#111111"),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

    if vlines:
        for x, label in vlines:
            ax.axvline(x=x, linestyle="--", linewidth=1.2, color="#777777", alpha=0.6)
            ax.text(
                x,
                0.02,
                str(label),
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.55),
            )

    for label, df in curves:
        is_highlight = label == highlight_label
        color = _color_for(label, default="#111111")
        ax.plot(
            df["dim"],
            df["accuracy"],
            marker=None,
            linewidth=2.6 if is_highlight else 1.2,
            color=color,
            alpha=1.0 if is_highlight else 0.25,
            label=label,
        )

        if is_highlight:
            # Annotate best point on highlighted curve
            best_idx = df["accuracy"].to_numpy().argmax()
            best_dim = float(df["dim"].iloc[best_idx])
            best_acc = float(df["accuracy"].iloc[best_idx])
            ax.scatter([best_dim], [best_acc], s=55, marker="*", color=color, zorder=5)
            ax.annotate(
                f"(d={int(best_dim)}, acc={best_acc:.3f})",
                xy=(best_dim, best_acc),
                xytext=(8, -10),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

    ax.set_xlabel("Dimension d")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.0, 1.0)

    # Keep legend small; icons still readable.
    ax.legend(fontsize=7, loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
