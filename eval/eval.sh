#!/bin/bash
# TopstepX 50K combine eval on real NQ 1s data (train split). Do not modify.
# Data: $NQ_DATA or ~/hive-data/nq/train.parquet. The holdout is scored separately by the task owner.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 eval/sim.py "$@" 2>&1 | tee run.log
