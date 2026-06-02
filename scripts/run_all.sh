#!/usr/bin/env bash
set -euo pipefail

bash scripts/train_pubmed.sh
bash scripts/train_wiki_cs.sh
bash scripts/train_photo.sh
bash scripts/train_ogbn_arxiv.sh
