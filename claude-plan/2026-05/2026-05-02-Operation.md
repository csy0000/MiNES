# Operation Guide — 2026-05-02

Steps still needed before the benchmark runs end-to-end.

## 1. Build the C++ binary

```bash
conda activate MiNES
clang++ -O2 -std=c++17 simulations/cpp/neq_sim.cpp -o simulations/cpp/neq_sim
./simulations/cpp/neq_sim --help
```

## 2. Smoke-test Python imports

```bash
conda activate MiNES
cd /Users/shuyuchen/Dropbox/ETH/Work/MiNES
python -c "
import sys
sys.path.insert(0, 'simulations')
sys.path.insert(0, 'src/analysis')
from adaptive_methods import build_grid
print('adaptive_methods OK')
"
python -c "
import sys
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'simulations')
sys.path.insert(0, 'src/analysis')
import mines_variance_fusion
print('mines_variance_fusion OK')
"
```

## 3. Add a `.gitignore`

Create `.gitignore` in the project root:

```
__pycache__/
*.pyc
*.pyo
data/
*.DS_Store
simulations/cpp/neq_sim
```

## 4. Run the 1D DoubleWell benchmark (smoke test)

Once the binary is built:

```bash
conda activate MiNES
RUN_AUS=0 RUN_MINES=0 RUN_NOTEBOOK=0 bash scripts/run_US_MTD_NES.sh
```

This runs only US, NES, and MTD (no adaptive methods) for a quick validation pass.
