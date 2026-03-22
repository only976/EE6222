# EE6222 Assignment 1

This folder contains a small, reusable experiment framework for:
- loading different high-dimensional datasets (MNIST, Olivetti faces)
- dimensionality reduction (none / PCA / LDA / PCA→LDA / PCA(>75%)+LDA / PCA(>90%)+LDA)
- classification (k-NN / Mahalanobis / Logistic Regression / Random Forest)
- sweeping feature dimension and plotting accuracy curves

## Setup

```bash
cd Assignment1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run a single experiment

Example (MNIST, PCA + 1-NN):

```bash
python run_experiment.py --dataset mnist --reducer pca --classifier knn --limit 20000
```

Example (faces, LDA + Mahalanobis):

```bash
python run_experiment.py --dataset olivetti_faces --reducer lda --classifier mahalanobis
```

Outputs:
- `outputs/<dataset>/<reducer>_<classifier>.csv`
- `outputs/<dataset>/<reducer>_<classifier>.png`

## Generate per-dataset curve tables (table + many curves)

This produces an Excel-like table (CSV) whose cells point to the generated curve images/CSVs.
It also generates an additional per-classifier comparison plot (multiple reducer curves overlaid in gray).

```bash
python run_summary.py --dataset mnist --limit 20000 --clean
python run_summary.py --dataset olivetti_faces --clean
```

Outputs:
- `outputs/<dataset>/curves/` (all per-combination curves as PNG+CSV)
- `outputs/<dataset>/curves/<classifier>/icon_summary.png` (a 5×1 icon-strip summary stored next to the raw curves)
- `outputs/<dataset>/compare/` (per-classifier gray overlaid comparison plots)
- `outputs/<dataset>/all_sweeps.csv` (long-form sweep results)
- `outputs/<dataset>/icon_grid_5x5.png` and `outputs/<dataset>/icon_rows/` (overlay icons; best per row is marked)
- `outputs/<dataset>/best_summary_heatmap.png` and `outputs/<dataset>/best_summary_table.csv` (best-accuracy heatmap)

Plot notes:
- PCA-related plots include dashed reference lines for the PCA dimensions that achieve ~90% and ~95% explained variance.

## Run the default set (both datasets)

```bash
python run_all.py
```

Notes:
- All preprocessing and model parameters are fit on the training split only.
- MNIST is downloaded via OpenML on first run; it may take a while.
