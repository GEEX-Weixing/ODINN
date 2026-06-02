# ODINN Anonymous Release

This is a cleaned, anonymous, runnable version of the ODINN project for paper review. It keeps only the core model, data loading, training/evaluation, and shell scripts needed to reproduce ODINN-DeGroot and ODINN-Friedkin-Johnsen experiments.

## Contents

```text
ODINN_clean/
├── models/                 # ODINN-DeGroot and ODINN-FJ implementations
├── src/                    # dataset loading, splitting, graph preprocessing, metrics
├── scripts/                # runnable experiment scripts
├── train.py                # train + validation + test evaluation
├── test.py                 # evaluate a saved checkpoint
├── requirements.txt
├── data/                   # empty placeholder; datasets download here
├── checkpoints/            # empty placeholder; checkpoints are saved here
└── logs/                   # empty placeholder; JSON/CSV logs are saved here
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA, install the PyTorch build that matches your CUDA version before installing the remaining packages.

## Supported datasets

The code supports:

- `pubmed` via `torch_geometric.datasets.Planetoid`
- `wiki_cs` via `torch_geometric.datasets.WikiCS`
- `photo` via `torch_geometric.datasets.Amazon(name="Photo")`
- `ogbn-arxiv` via `ogb.nodeproppred.PygNodePropPredDataset`

Datasets are not included in this anonymous package. They are downloaded and cached under `data/` automatically by PyG/OGB.

## Train

Run one experiment manually:

```bash
python train.py --dataset pubmed --model_name DeGroot --label_rate 1 --num_layers 10 --epochs 400 --folds 10
python train.py --dataset pubmed --model_name Friedkin_Johnsen --label_rate 1 --num_layers 10 --epochs 400 --folds 10
```

Or use the provided scripts:

```bash
bash scripts/train_pubmed.sh
bash scripts/train_wiki_cs.sh
bash scripts/train_photo.sh
bash scripts/train_ogbn_arxiv.sh
```

Run all four dataset scripts:

```bash
bash scripts/run_all.sh
```

## Splits

By default, `--split generated` creates stratified few-shot splits. `--label_rate` is the number of labeled nodes sampled per class; fractional legacy values are rounded up to at least one sample per class.

For `ogbn-arxiv`, you can use the OGB official split instead:

```bash
python train.py --dataset ogbn-arxiv --split official --model_name DeGroot --num_layers 20
```

## Test a saved checkpoint

```bash
python test.py --checkpoint checkpoints/pubmed_DeGroot_fold0.pt
```

The checkpoint stores the model hyperparameters and split metadata used by `train.py`, so `test.py` can reconstruct the model and split automatically.

## Notes on cleanup

Removed from the original working directory: IDE files, macOS artifacts, Python bytecode caches, local dataset caches, duplicated scripts, and baseline script references without corresponding implementation. The release contains no author names, personal paths, local logs, or trained weights.
