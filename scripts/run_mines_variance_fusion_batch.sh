#!/usr/bin/env bash

# Use your exact system root; from earlier yours includes kT_1p0:
SYSTEM_ROOT="data/1D/DoubleWell__k0_1p0__x0_m10p0__k1_1p0__x1_10p0__E1_10p0__kT_1p0__dt_0p0005__gamma_1p0"
mkdir -p logs # Ensure logs directory exists
for T_NEQ in 1000 2000 3000 ; do
  bash scripts/run_mines_variance_fusion.sh \
    --system-root "${SYSTEM_ROOT}" \
    --t-neq "${T_NEQ}" \
    --label "mines_variance_fusion_t${T_NEQ}" \
    --pmf-method hybrid \
    --js-threshold 0.8 > "logs/mines_variance_fusion_t${T_NEQ}.log" 2>&1 
done
