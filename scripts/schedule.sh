#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=symbolic_probe data.batch_size=64
