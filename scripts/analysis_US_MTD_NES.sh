#!/usr/bin/env bash
set -euo pipefail

# This wrapper finalizes the streamed 1D DoubleWell benchmark outputs stored
# under data/.
# - raw method blocks should already have been reduced seed-by-seed by the main
#   runner
# - this pass aggregates seed-level PMF files, ranks screened US / AUS / NES /
#   MINES / WT-MTD combinations, and writes the stable benchmark-facing files
#   under benchmark/selected
#
# Positional parameter:
# - $1 optional system root; defaults to the active DoubleWell system under data/1D
# The reducer reads run_context.json from that system root and finalizes the
# method-level benchmark outputs.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

slug_float() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "$value"
}

DEFAULT_SYSTEM_ROOT=$(
  printf '%s/data/1D/DoubleWell__k0_%s__x0_%s__k1_%s__x1_%s__E1_%s__kT_%s__dt_%s__gamma_%s' \
    "$ROOT_DIR" \
    "$(slug_float 1.0)" \
    "$(slug_float -10.0)" \
    "$(slug_float 1.0)" \
    "$(slug_float 10.0)" \
    "$(slug_float 10.0)" \
    "$(slug_float 1.0)" \
    "$(slug_float 0.0005)" \
    "$(slug_float 1.0)"
)
SYSTEM_ROOT="${1:-$DEFAULT_SYSTEM_ROOT}"

python3 "$ROOT_DIR/src/analysis/analysis_US_MTD.py" \
  finalize \
  --system-root "$SYSTEM_ROOT"

echo "Wrote analyzed benchmark files under $SYSTEM_ROOT"
