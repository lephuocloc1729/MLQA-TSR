#!/bin/bash
set -euo pipefail

include_public=0
include_private=0

usage() {
  cat <<'EOF'
Usage: scripts/check_data.sh [--public] [--private] [--all]

Checks local raw-data placement for the course prototype.
Required by default: LawDB and train split inputs.
Optional flags: public/private test inputs for submission rehearsal.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --public)
      include_public=1
      ;;
    --private)
      include_private=1
      ;;
    --all)
      include_public=1
      include_private=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

missing=0

check_file() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]]; then
    echo "OK file: $label -> $path"
  else
    echo "Missing file: $label -> $path" >&2
    missing=1
  fi
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "$path" ]]; then
    echo "OK dir:  $label -> $path"
  else
    echo "Missing dir:  $label -> $path" >&2
    missing=1
  fi
}

echo "Checking required dataset files..."
check_file "data/raw/law_db/vlsp2025_law_new.json" "LawDB JSON"
check_dir "data/raw/law_db/images.fld" "LawDB images"
check_file "data/raw/train_data/vlsp_2025_train.json" "train JSON"
check_dir "data/raw/train_data/train_images" "train images"

if [[ "$include_public" -eq 1 ]]; then
  echo "Checking public test files..."
  check_file "data/raw/public_test/vlsp_2025_public_test_task1.json" "public task 1 JSON"
  check_file "data/raw/public_test/vlsp_2025_public_test_task2.json" "public task 2 JSON"
  check_dir "data/raw/public_test/public_test_images" "public test images"
fi

if [[ "$include_private" -eq 1 ]]; then
  echo "Checking private test files..."
  check_file "data/raw/private_test/Task 1 Submission File/vlsp2025_submission_task1.json" "private task 1 JSON"
  check_file "data/raw/private_test/Task 2 Submission File/vlsp2025_submission_task2.json" "private task 2 JSON"
  check_dir "data/raw/private_test/private_test_images" "private test images"
fi

if [[ "$missing" -ne 0 ]]; then
  echo "Data check failed. Place the missing files under data/raw/ before release rehearsal." >&2
  exit 1
fi

echo "Data check passed."
