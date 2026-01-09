#!/bin/bash
set -euo pipefail

# This bash script demonstrates how to invoke the Python scripts related to scalability_randomtopo experiment.
# If invoked as is, the whole process would take multiple days.
# To speed up the evaluation, you can arrange scalability_randomtopo_run.py to run in parallel
# with proper CPU isolation, and then run scalability_randomtopo_plot.py to plot the diagrams.

OUTDIR=examples/scalability_randomtopo
mkdir -p $OUTDIR
echo '*' >$OUTDIR/.gitignore

RUNS=5
SD=3.0
QC=10
TL=10800
ENABLE_SEQUENCE=1

run_seeds() {
  local SEED_BASE=200
  for SEED in $(seq $SEED_BASE $((SEED_BASE+RUNS-1))); do
    python $1 --seed $SEED --nnodes $2 --nedges $3 --sim_duration $SD --qchannel_capacity $QC --time_limit $TL --outdir $OUTDIR
  done
}

run_simulator() {
  run_seeds $1 16  20
  run_seeds $1 32  40
  run_seeds $1 64  80
  run_seeds $1 128 160
  run_seeds $1 256 320
  run_seeds $1 512 640
}

run_simulator examples/scalability_randomtopo_run.py
if [[ $ENABLE_SEQUENCE -ne 0 ]]; then
  run_simulator examples/sequence/scalability_randomtopo_run.py
  SEQUENCE_FLAG=--sequence
else
  SEQUENCE_FLAG=''
fi

python examples/scalability_randomtopo_plot.py \
  --indir $OUTDIR $SEQUENCE_FLAG \
  --runs $RUNS --qchannel_capacity $QC --time_limit $TL \
  --csv $OUTDIR/$QC.csv --plt $OUTDIR/$QC.png
