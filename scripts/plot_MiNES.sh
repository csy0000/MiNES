#!/usr/bin/env bash
set -euo pipefail

# Read the packaged MiNES analysis bundle and write a compact overview figure
# under analysis/.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
MPL_CONFIG_DIR="$ROOT_DIR/.matplotlib-cache"
mkdir -p "$MPL_CONFIG_DIR"
export MPLCONFIGDIR="$MPL_CONFIG_DIR"

BUNDLE_PATH="${MINES_BUNDLE_PATH:-$ROOT_DIR/analysis/MiNES/${MINES_OUTPUT_STEM:-mines_current_protocol_t5000_n50}.pkl}"
OUTPUT_DIR="${MINES_PLOT_DIR:-$ROOT_DIR/analysis/MiNES/${MINES_OUTPUT_STEM:-mines_current_protocol_t5000_n50}}"

if [[ ! -f "$BUNDLE_PATH" ]]; then
  echo "MiNES analysis bundle not found: $BUNDLE_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 - "$BUNDLE_PATH" "$OUTPUT_DIR" <<'PY'
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

bundle_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

with bundle_path.open("rb") as handle:
    bundle = pickle.load(handle)

figure_paths = bundle["figure_paths"]
figure_keys = ["figure_1", "figure_2", "figure_3", "figure_4"]
images = []
for key in figure_keys:
    path = Path(figure_paths[key])
    if not path.exists():
        raise FileNotFoundError(f"Missing figure for MiNES overview: {path}")
    images.append((key, mpimg.imread(path)))

metrics = bundle.get("metrics", {})
coverage_metrics = metrics.get("coverage_metrics", [])
coverage_ratio = float(coverage_metrics[-1]["coverage_ratio"]) if coverage_metrics else float("nan")
stop_reason = str(metrics.get("stop_reason", ""))

fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
for ax, (key, image) in zip(axes.flat, images):
    ax.imshow(image)
    ax.set_title(key.replace("_", " ").title())
    ax.axis("off")

fig.suptitle(
    f"MiNES {bundle['label']} | stop={stop_reason} | coverage={coverage_ratio:.3f}",
    fontsize=14,
)

overview_path = output_dir / "MiNES_overview.png"
fig.savefig(overview_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(str(overview_path))
PY
