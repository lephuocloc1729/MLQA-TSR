#!/bin/bash

echo "Checking dataset files..."

test -f data/raw/law_db/vlsp2025_law_new.json || echo "Missing LawDB JSON"
test -d data/raw/law_db/images.fld || echo "Missing LawDB images"
test -f data/raw/train/vlsp_2025_train.json || echo "Missing train JSON"
test -d data/raw/train/train_images || echo "Missing train images"

echo "Done."