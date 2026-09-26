# ODINN Few-Shot Reproduction

Minimal release containing only the ODINN model and the few-shot node-classification experiments used in the paper.

## Included

- **ODINN-DG** and **ODINN-FJ** implementations.
- **Homophilous few-shot experiments** on Cora, Pubmed, Amazon Computers, Amazon Photo, Coauthor CS, Coauthor Physics, Wiki-CS, and OGBN-Arxiv.
- **Heterophilous few-shot experiments** on Chameleon, Squirrel, Amazon-Ratings, and Penn94.
- The paper setting of **1/3/5 labeled nodes per class**, **30 random splits**, and candidate propagation depths **K = 10/20/40**.

No baseline implementations, sensitivity studies, learned-weight visualization, runtime experiments, long-term dynamics experiments, high-shot sweeps, or ablation scripts are included.

## Structure

```text
ODINN_fewshot_release/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── src/
│   ├── odinn.py
│   ├── odinn_heterophily.py
│   ├── odinn_weight_init.py
│   ├── graph_ops.py
│   ├── training.py
│   ├── data.py
│   ├── data_heterophily.py
│   └── io_utils.py
├── experiments/
│   ├── fewshot_homophily.py
│   └── fewshot_heterophily.py
└── scripts/
    ├── run_fewshot_homophily.sh
    └── run_fewshot_heterophily.sh
```

## Installation

```bash
pip install -r requirements.txt
```

Prepare the datasets under `data/` first.

## Run

Homophilous graphs:

```bash
bash scripts/run_fewshot_homophily.sh
```

Heterophilous graphs:

```bash
bash scripts/run_fewshot_heterophily.sh
```

You can pass additional arguments through the shell scripts, for example:

```bash
bash scripts/run_fewshot_homophily.sh --datasets cora pubmed --models dg fj --cpu
bash scripts/run_fewshot_heterophily.sh --datasets chameleon penn94 --cpu
```

Each experiment writes all runs to `results/` and produces a summary in which the propagation depth is selected using mean validation accuracy over the 30 splits.
