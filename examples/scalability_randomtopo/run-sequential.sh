#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PARAMS=${1:-params-full.toml}
eval "$(python srt_params_bash.py --params "$PARAMS")"

OUTDIR=${2:-output}
mkdir -p $OUTDIR
echo '*' >$OUTDIR/.gitignore
cp "$PARAMS" "$OUTDIR"/params.toml

run_seeds() {
  local SCRIPT=$1
  local NODES=$2
  local EDGES=$3
  for SEED in $(seq $SEED_BASE $((SEED_BASE+RUNS-1))); do
    python $SCRIPT --params "$PARAMS" --seed $SEED --nodes $NODES --edges $EDGES --outdir "$OUTDIR"
  done
}

run_simulator() {
  local SCRIPT=$1
  for I in "${!NS_NODES[@]}"; do
    run_seeds $SCRIPT "${NS_NODES[$I]}" "${NS_EDGES[$I]}"
  done
}

run_simulator srt_mqns.py
if [[ $ENABLE_SEQUENCE -ne 0 ]]; then
  run_simulator srt_sequence.py
  PLOT_FLAG=--sequence
else
  PLOT_FLAG=''
fi

python srt_plot.py \
  --params "$OUTDIR"/params.toml --indir "$OUTDIR" $PLOT_FLAG \
  --csv "$OUTDIR/srt.csv" --plt "$OUTDIR/srt.png"
