#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets import load_dataset
from src.experiment import ExperimentConfig, run_dimension_sweep, save_results
from src.plotting import plot_accuracy_curve


def main():
    p = argparse.ArgumentParser(description="EE6222 Assignment1: DR + classification sweep")
    p.add_argument("--dataset", choices=["mnist", "olivetti_faces"], required=True)
    p.add_argument("--limit", type=int, default=None, help="Optional sub-sample size")
    p.add_argument(
        "--reducer",
        choices=["none", "pca", "lda", "pca_lda", "pca75_lda", "pca90_lda", "pca95_lda", "pca99_lda"],
        default="pca",
    )
    p.add_argument(
        "--classifier",
        choices=["knn", "mahalanobis", "logreg", "rf"],
        default="knn",
    )
    p.add_argument("--knn-k", type=int, default=1)
    p.add_argument("--rf-n-estimators", type=int, default=200)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Output directory for CSV and plots",
    )

    args = p.parse_args()

    dataset = load_dataset(args.dataset, limit=args.limit, random_state=args.seed)
    cfg = ExperimentConfig(
        reducer=args.reducer,
        classifier=args.classifier,
        test_size=args.test_size,
        random_state=args.seed,
        standardize=not args.no_standardize,
        knn_k=args.knn_k,
        rf_n_estimators=args.rf_n_estimators,
    )

    df = run_dimension_sweep(dataset, cfg)

    tag = f"{args.reducer}_{args.classifier}"
    out_csv = args.outdir / args.dataset / f"{tag}.csv"
    out_png = args.outdir / args.dataset / f"{tag}.png"

    save_results(df, out_csv)
    plot_accuracy_curve(
        df,
        title=f"{args.dataset}: {args.reducer} + {args.classifier}",
        out_path=out_png,
    )

    best = df.iloc[df["accuracy"].to_numpy().argmax()]
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")
    print(f"Best: dim={int(best['dim'])}, acc={float(best['accuracy']):.4f}")


if __name__ == "__main__":
    main()
