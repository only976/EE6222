#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from src.datasets import load_dataset
from src.experiment import ExperimentConfig, run_dimension_sweep, save_results
from src.plotting import plot_accuracy_curve, plot_overlaid_curves, plot_overlaid_curves_highlight_one


def _method_specs():
    # label -> reducer
    return [
        ("Raw Data", "none"),
        ("PCA", "pca"),
        ("LDA", "lda"),
        ("PCA (>90%) +LDA", "pca90_lda"),
        ("PCA (>95%) +LDA", "pca95_lda"),
        ("PCA (>99%) +LDA", "pca99_lda"),
    ]


def _method_colors(method_labels: list[str]) -> dict[str, str]:
    # Stable label->color mapping (so reordering methods doesn't change colors).
    fixed = {
        "Raw Data": "#000000",
        "PCA": "#1f77b4",
        "LDA": "#ff7f0e",
        "PCA (>90%) +LDA": "#d62728",
        "PCA (>95%) +LDA": "#2ca02c",
        "PCA (>99%) +LDA": "#9467bd",
    }

    # Fall back to a small palette if unexpected labels appear.
    fallback_palette = [
        "#000000",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    colors: dict[str, str] = {}
    next_idx = 0
    for label in method_labels:
        if label in fixed:
            colors[label] = fixed[label]
        else:
            colors[label] = fallback_palette[next_idx % len(fallback_palette)]
            next_idx += 1
    return colors


def _classifier_specs(rf_n_estimators: int):
    # row label -> (classifier, knn_k)
    return [
        ("The minimum Mahalanobis distance classifier", ("mahalanobis", None)),
        ("linear classifier", ("logreg", None)),
        ("KNN-1", ("knn", 1)),
        ("KNN-5", ("knn", 5)),
        ("Random Forest", ("rf", None)),
    ]


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "x"


def plot_curve_thumbnail_grid(
    *,
    png_table: pd.DataFrame,
    best_method_per_classifier: dict[str, str] | None,
    out_path: Path,
    title: str,
    gray_non_best: bool = True,
):
    """Create a grid image where each cell shows a curve PNG thumbnail.

        If `best_method_per_classifier` is provided:
            - highlight the best method cell per row
            - gray-out other methods within the same row
        If it is None:
            - just render the provided images as-is (useful when the images are already
                "single-highlight" icons)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.patches import Rectangle

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep visuals stable and avoid style-induced gridlines.
    try:
        plt.style.use("default")
    except Exception:
        pass

    n_rows, n_cols = png_table.shape
    # Composite images tend to get hard to read when embedded in reports; render them large.
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4.2, n_rows * 3.0),
        dpi=220,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(title, y=0.995, fontsize=12)

    # Column headers
    for j, col in enumerate(png_table.columns):
        axes[0, j].set_title(str(col), fontsize=10, pad=10)

    for i, clf_label in enumerate(png_table.index):
        best_method = None
        if best_method_per_classifier is not None:
            best_method = best_method_per_classifier.get(str(clf_label), None)

        for j, method_label in enumerate(png_table.columns):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            rel_path = png_table.loc[clf_label, method_label]
            if not isinstance(rel_path, str) or not rel_path:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center", fontsize=9)
                continue

            img = mpimg.imread(rel_path)
            is_best = (best_method is not None) and (str(method_label) == best_method)
            if best_method_per_classifier is not None and gray_non_best and not is_best:
                # If this cell is not best in the row, render it grayscale.
                if img.ndim == 3 and img.shape[2] >= 3:
                    rgb = img[..., :3]
                    gray = np.dot(rgb, [0.299, 0.587, 0.114])
                    img = np.stack([gray, gray, gray], axis=-1)

            ax.imshow(img, interpolation="nearest")

            # Row label at the left of first column
            if j == 0:
                ax.set_ylabel(str(clf_label), rotation=0, ha="right", va="center", fontsize=10, labelpad=50)

            # Highlight best cell with a thick border (only in best-mode)
            if best_method_per_classifier is not None and is_best:
                ax.add_patch(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=ax.transAxes,
                        fill=False,
                        linewidth=3.0,
                        edgecolor="#000000",
                    )
                )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_best_accuracy_heatmap(summary: pd.DataFrame, *, out_path: Path, title: str):
    import numpy as np
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep style stable and avoid style-induced gridlines.
    try:
        plt.style.use("default")
    except Exception:
        pass

    # Nice defaults for macOS; safe if missing.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    data = summary.to_numpy(dtype=float)

    # Make room for long classifier labels.
    fig, ax = plt.subplots(figsize=(12.5, 3.6), dpi=240)
    # Red (low) -> Green (high)
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(
        data,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
        resample=False,
    )

    ax.set_xticks(np.arange(summary.shape[1]))
    ax.set_yticks(np.arange(summary.shape[0]))
    ax.set_xticklabels(summary.columns, rotation=20, ha="right", fontsize=9)

    def _wrap_label(s: str, max_len: int = 28) -> str:
        s = str(s)
        if len(s) <= max_len:
            return s
        parts = s.split(" ")
        lines: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for p in parts:
            add = len(p) + (1 if cur else 0)
            if cur_len + add > max_len and cur:
                lines.append(" ".join(cur))
                cur = [p]
                cur_len = len(p)
            else:
                cur.append(p)
                cur_len += add
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines[:2])  # keep it compact

    ax.set_yticklabels([_wrap_label(x) for x in summary.index], fontsize=10)
    ax.set_title(title)
    ax.grid(False)

    # Annotate values with auto-contrast text color.
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    for i in range(summary.shape[0]):
        for j in range(summary.shape[1]):
            v = float(data[i, j])
            r, g, b, _ = cmap(norm(v))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            txt_color = "black" if luminance > 0.6 else "white"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=txt_color)

    # No best-cell highlighting; user can infer visually.

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Best accuracy")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=(
            "Generate per-dataset curve artifacts (CSV+PNG) for each (classifier, method), "
            "and output a table CSV that maps cells to those curve files."
        )
    )
    p.add_argument("--dataset", choices=["mnist", "olivetti_faces"], required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rf-n-estimators", type=int, default=200)
    p.add_argument("--clean", action="store_true", help="Remove outputs/<dataset>/ before running")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Output directory for CSV and plots",
    )

    args = p.parse_args()

    dataset = load_dataset(args.dataset, limit=args.limit, random_state=args.seed)

    method_specs = _method_specs()
    classifier_specs = _classifier_specs(args.rf_n_estimators)
    method_labels = [m for m, _ in method_specs]
    method_colors = _method_colors(method_labels)

    out_ds = args.outdir / args.dataset
    if args.clean and out_ds.exists():
        shutil.rmtree(out_ds)
    out_curves = out_ds / "curves"
    out_ds.mkdir(parents=True, exist_ok=True)

    classifier_labels = [clf_label for clf_label, _ in classifier_specs]
    method_labels = [m for m, _ in method_specs]

    # For generating per-classifier overlay plots.
    per_classifier_curves: dict[str, list[tuple[str, pd.DataFrame]]] = {
        clf_label: [] for clf_label, _ in classifier_specs
    }
    per_classifier_baseline: dict[str, float] = {}

    # Track curves by method label (for icon generation).
    per_classifier_by_method: dict[str, dict[str, pd.DataFrame]] = {
        clf_label: {} for clf_label, _ in classifier_specs
    }

    pca_curves: dict[str, pd.DataFrame] = {}

    all_sweeps: list[pd.DataFrame] = []

    total_jobs = len(classifier_specs) * len(method_specs)
    done_jobs = 0

    for clf_label, (clf_name, knn_k) in classifier_specs:
        clf_key = _slug(clf_label)

        for method_label, reducer in method_specs:
            done_jobs += 1
            print(f"[{args.dataset}] ({done_jobs}/{total_jobs}) {clf_label} / {method_label} ...")
            method_key = _slug(method_label)

            cfg = ExperimentConfig(
                reducer=reducer,  # type: ignore[arg-type]
                classifier=clf_name,  # type: ignore[arg-type]
                test_size=args.test_size,
                random_state=args.seed,
                standardize=True,
                knn_k=int(knn_k) if knn_k is not None else 1,
                rf_n_estimators=args.rf_n_estimators,
            )

            df = run_dimension_sweep(dataset, cfg)
            df = df.copy()
            df["method_label"] = method_label
            df["classifier_label"] = clf_label
            all_sweeps.append(df)

            # Save per-combination curve artifacts
            out_dir = out_curves / clf_key
            out_csv = out_dir / f"{method_key}.csv"
            out_png = out_dir / f"{method_key}.png"
            save_results(df, out_csv)
            plot_accuracy_curve(df, title=f"{args.dataset}: {clf_label} / {method_label}", out_path=out_png)

            png_rel = str(out_png.relative_to(args.outdir))
            csv_rel = str(out_csv.relative_to(args.outdir))
            _ = png_rel, csv_rel  # keep for potential future tables

            if reducer == "none":
                per_classifier_baseline[clf_label] = float(df["accuracy"].iloc[0])
            else:
                per_classifier_curves[clf_label].append((method_label, df[["dim", "accuracy"]]))

            per_classifier_by_method[clf_label][method_label] = df[["dim", "accuracy"]]

            if reducer == "pca":
                pca_curves[clf_label] = df

    # Save the tables and the full long-form sweeps.
    all_sweeps_df = pd.concat(all_sweeps, ignore_index=True)
    out_all = out_ds / "all_sweeps.csv"
    save_results(all_sweeps_df, out_all)

    # Compute best method per classifier (for marking in 5x5/5x1 grids)
    best_df = all_sweeps_df.groupby(["classifier_label", "method_label"], as_index=False)["accuracy"].max()
    best_summary = best_df.pivot(index="classifier_label", columns="method_label", values="accuracy")
    best_summary = best_summary.loc[classifier_labels, method_labels]
    best_method_per_classifier: dict[str, str] = (
        best_summary.idxmax(axis=1).astype(str).to_dict()  # type: ignore[assignment]
    )

    # PCA reference dimensions for 90%/95% explained variance (from the PCA+LDA runs)
    vlines: list[tuple[float, str]] = []
    try:
        d90 = int(
            all_sweeps_df.loc[all_sweeps_df["reducer"] == "pca90_lda", "pca_dim"]
            .dropna()
            .astype(int)
            .iloc[0]
        )
        vlines.append((float(d90), f"90% (d={d90})"))
    except Exception:
        pass
    try:
        d95 = int(
            all_sweeps_df.loc[all_sweeps_df["reducer"] == "pca95_lda", "pca_dim"]
            .dropna()
            .astype(int)
            .iloc[0]
        )
        vlines.append((float(d95), f"95% (d={d95})"))
    except Exception:
        pass

    try:
        d99 = int(
            all_sweeps_df.loc[all_sweeps_df["reducer"] == "pca99_lda", "pca_dim"]
            .dropna()
            .astype(int)
            .iloc[0]
        )
        vlines.append((float(d99), f"99% (d={d99})"))
    except Exception:
        pass

    # Re-plot PCA single-condition curves with 90%/95% reference lines.
    if vlines:
        for clf_label, df_pca in pca_curves.items():
            clf_key = _slug(clf_label)
            out_png = out_curves / clf_key / "pca.png"
            plot_accuracy_curve(
                df_pca,
                title=f"{args.dataset}: {clf_label} / PCA",
                out_path=out_png,
                vlines=vlines,
            )

    # Per-classifier overlay comparison plots (gray curves)
    out_cmp = out_ds / "compare"
    compare_rows: list[dict] = []
    for clf_label in classifier_labels:
        curves = per_classifier_curves.get(clf_label, [])
        baseline_acc = per_classifier_baseline.get(clf_label)
        baseline = ("Raw Data", baseline_acc) if baseline_acc is not None else None
        if not curves:
            continue
        out_path = out_cmp / f"{_slug(clf_label)}.png"
        plot_overlaid_curves(
            curves,
            title=f"{args.dataset}: {clf_label} (reducers comparison)",
            out_path=out_path,
            baseline=baseline,
            vlines=vlines if vlines else None,
        )

        compare_rows.append(
            {
                "classifier": clf_label,
                "compare_png": str(out_path.relative_to(args.outdir)),
            }
        )

    out_compare_table = out_ds / "compare_table.csv"
    pd.DataFrame(compare_rows).to_csv(out_compare_table, index=False)

    # Icon grid: each cell is an overlay plot with only the column's method highlighted.
    out_icons = out_ds / "icons"
    icon_png_table = pd.DataFrame(
        index=[clf_label for clf_label, _ in classifier_specs],
        columns=[m for m, _ in method_specs],
        dtype=object,
    )
    icon_png_table_abs = icon_png_table.copy()

    for clf_label in icon_png_table.index:
        curves = per_classifier_curves.get(clf_label, [])
        baseline_acc = per_classifier_baseline.get(clf_label)
        baseline = ("Raw Data", baseline_acc) if baseline_acc is not None else None
        clf_dir = out_icons / _slug(clf_label)

        for method_label in icon_png_table.columns:
            out_icon = clf_dir / f"{_slug(str(method_label))}.png"
            plot_overlaid_curves_highlight_one(
                curves,
                title=f"{args.dataset}: {clf_label}",
                out_path=out_icon,
                highlight_label=str(method_label),
                baseline=baseline,
                method_colors=method_colors,
                vlines=vlines if vlines else None,
            )
            icon_png_table.loc[clf_label, method_label] = str(out_icon.relative_to(args.outdir))
            icon_png_table_abs.loc[clf_label, method_label] = str(out_icon)

    out_icon_table = out_ds / "icon_table_png.csv"
    icon_png_table.to_csv(out_icon_table, index=True)

    out_grid = out_ds / "icon_grid_5x5.png"
    plot_curve_thumbnail_grid(
        png_table=icon_png_table_abs,
        best_method_per_classifier=best_method_per_classifier,
        out_path=out_grid,
        title=f"{args.dataset}: icon grid (column method highlighted)",
        gray_non_best=False,
    )

    # 5x1 icon strip per classifier
    out_rows = out_ds / "icon_rows"
    row_rows: list[dict] = []
    for clf_label in icon_png_table_abs.index:
        row_df = pd.DataFrame([icon_png_table_abs.loc[clf_label].to_list()], columns=icon_png_table_abs.columns)
        row_df.index = [clf_label]
        out_row_png = out_rows / f"{_slug(str(clf_label))}.png"
        plot_curve_thumbnail_grid(
            png_table=row_df,
            best_method_per_classifier=best_method_per_classifier,
            out_path=out_row_png,
            title=f"{args.dataset}: {clf_label} (icons)",
            gray_non_best=False,
        )
        row_rows.append({"classifier": clf_label, "icon_row_png": str(out_row_png.relative_to(args.outdir))})

    out_row_table = out_ds / "icon_row_table.csv"
    pd.DataFrame(row_rows).to_csv(out_row_table, index=False)

    # Add one icon-summary strip inside each per-classifier curve folder.
    # This duplicates the 5×1 icon strip but stores it alongside the raw per-method curves.
    for clf_label in icon_png_table_abs.index:
        row_df = pd.DataFrame([icon_png_table_abs.loc[clf_label].to_list()], columns=icon_png_table_abs.columns)
        row_df.index = [clf_label]
        out_icon_summary = out_curves / _slug(str(clf_label)) / "icon_summary.png"
        plot_curve_thumbnail_grid(
            png_table=row_df,
            best_method_per_classifier=best_method_per_classifier,
            out_path=out_icon_summary,
            title=f"{args.dataset}: {clf_label} (icon summary)",
            gray_non_best=False,
        )

    # 5x5 best-accuracy heatmap (numbers)
    out_best_table = out_ds / "best_summary_table.csv"
    out_best_long = out_ds / "best_summary_long.csv"
    out_best_png = out_ds / "best_summary_heatmap.png"
    best_summary.to_csv(out_best_table, index=True)
    best_df.rename(columns={"classifier_label": "classifier", "method_label": "method", "accuracy": "best_accuracy"}).to_csv(
        out_best_long, index=False
    )
    plot_best_accuracy_heatmap(best_summary, out_path=out_best_png, title=f"{args.dataset}: best accuracy heatmap")

    print(f"Saved: {out_all}")
    print(f"Saved: {out_cmp}")
    print(f"Saved: {out_compare_table}")
    print(f"Saved: {out_icon_table}")
    print(f"Saved: {out_grid}")
    print(f"Saved: {out_row_table}")
    print(f"Saved: {out_best_table}")
    print(f"Saved: {out_best_png}")


if __name__ == "__main__":
    main()
