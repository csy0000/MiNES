#!/usr/bin/env bash
# Run only the Adaptive Umbrella Sampling (AUS) benchmark arm.
# All parameters are forwarded to run_US_MTD_NES.sh.
# Accepts the same environment variable overrides (POT_*, AUS_*, SEEDS_CSV, etc.).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

RUN_US=0 RUN_AUS=1 RUN_NES=0 RUN_MINES=0 RUN_MTD=0 RUN_PLOTS=0 RUN_NOTEBOOK=0 \
  bash "$SCRIPT_DIR/run_US_MTD_NES.sh" "$@"
