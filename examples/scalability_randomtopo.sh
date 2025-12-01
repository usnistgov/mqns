#!/bin/bash
set -euo pipefail

# The scalability_randomtopo scripts measure how simulation performance and outcomes scale as the
# network size increases. A random topology is used with an average node degree of 2.5.
# For each network size, the number of entanglement requests is chosen to be proportional to the
# number of nodes, with 20% of nodes involved in src-dst requests (plus intermediate nodes).
# Proactive forwarding is used with Statistical multiplexing and SWAP-ASAP swapping policy.
# The simulation reports execution time.
#
# This bash script demonstrates how to invoke the Python scripts.
# If invoked as is, the whole process would take more than 24 hours.
# To speed up the evaluation, you can arrange scalability_randomtopo_run.py to run in parallel
# with proper CPU isolation, and then run scalability_randomtopo_plot.py to plot the diagrams.

OUTDIR=examples/scalability_randomtopo/
mkdir -p $OUTDIR
echo '*' >$OUTDIR/.gitignore

RUNS=5
QC=10

run_network_size() {
  for SEED in $(seq 200 $((200+RUNS))); do
    python examples/scalability_randomtopo_run.py --seed $SEED --nnodes $1 --nedges $2 --qchannel_capacity $QC --outdir $OUTDIR
  done
}

run_network_size 16  20
run_network_size 32  40
run_network_size 64  80
run_network_size 128 160
run_network_size 256 320
run_network_size 512 640

python examples/scalability_randomtopo_plot.py --indir $OUTDIR --runs $RUNS --qchannel_capacity $QC --csv $OUTDIR/$QC.csv --plt $OUTDIR/$QC.png
