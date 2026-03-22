#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from src.datasets import load_dataset
from src.experiment import ExperimentConfig, run_dimension_sweep, save_results
from src.plotting import plot_accuracy_curve


def run_one(dataset_name: str, *, limit: int | None, outdir: Path, seed: int):
    dataset = load_dataset(dataset_name, limit=limit, random_state=seed)

    configs = [
        ExperimentConfig(reducer="none", classifier="knn", random_state=seed),
        ExperimentConfig(reducer="pca", classifier="knn", random_state=seed),
        ExperimentConfig(reducer="lda", classifier="knn", random_state=seed),
        ExperimentConfig(reducer="pca_lda", classifier="knn", random_state=seed),
    ]

    for cfg in configs:
        df = run_dimension_sweep(dataset, cfg)
        tag = f"{cfg.reducer}_{cfg.classifier}"
        out_csv = outdir / dataset_name / f"{tag}.csv"
        out_png = outdir / dataset_name / f"{tag}.png"
        save_results(df, out_csv)
        plot_accuracy_curve(df, title=f"{dataset_name}: {tag}", out_path=out_png)
        best = df.iloc[df["accuracy"].to_numpy().argmax()]
        print(f"{dataset_name} {tag}: best dim={int(best['dim'])}, acc={float(best['accuracy']):.4f}")


def main():
    outdir = Path("outputs")
    seed = 0

    # MNIST can be heavy on first download. Limit makes iteration faster.
    run_one("mnist", limit=20000, outdir=outdir, seed=seed)
    run_one("olivetti_faces", limit=None, outdir=outdir, seed=seed)


if __name__ == "__main__":
    main()
