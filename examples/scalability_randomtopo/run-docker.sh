#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PARAMS=${1:-params-full.toml}
eval "$(python srt_params_bash.py --params "$PARAMS")"

if [[ $RUNS -gt "${#CPUSET_CPUS[@]}" ]]; then
  echo "$PARAMS: \`cpuset_cpus\` has fewer than \`runs\` cores"
  exit 1
fi

OUTDIR=${2:-output}
mkdir -p $OUTDIR
echo '*' >$OUTDIR/.gitignore
cp "$PARAMS" "$OUTDIR"/params.toml

mkdir -p docker.local
echo '*' >docker.local/.gitignore
DOCKER_RUN_ARGS="-u $(id -u):$(id -g) -v $(readlink -f ../..):/app -v $(readlink -f docker.local):/home/user/.local -v $(readlink -f $OUTDIR):/output -w /app --env HOME=/home/user"

if ! [[ -f docker.local/DONE ]]; then
  docker run --rm $DOCKER_RUN_ARGS python:3.12 bash -c '
    pip install --user --no-warn-script-location -r requirements.txt
    pip install --user --no-warn-script-location -e .
    pip install --user --no-warn-script-location -r examples/scalability_randomtopo/requirements.txt
    touch ~/.local/DONE
  '
fi

CT_CMD='cd examples/scalability_randomtopo'

append_cmd() {
  CT_CMD="$CT_CMD; $1"
}

plan_simulator() {
  local SCRIPT=$1
  for I in "${!NS_NODES[@]}"; do
    append_cmd "python $SCRIPT --params /output/params.toml --seed \$SEED --nodes ${NS_NODES[$I]} --edges ${NS_EDGES[$I]} --outdir /output"
  done
}

plan_simulator srt_mqns.py
if [[ $ENABLE_SEQUENCE -ne 0 ]]; then
  plan_simulator srt_sequence.py
  PLOT_FLAG=--sequence
else
  PLOT_FLAG=''
fi

for I in $(seq 0 $((RUNS-1))); do
  CT=mqns_srt_$I
  docker run -d $DOCKER_RUN_ARGS --name $CT --env SEED=$((SEED_BASE+I)) --cpuset-cpus "${CPUSET_CPUS[$I]}" --network none python:3.12 bash -c "$CT_CMD"
done

for I in $(seq 0 $((RUNS-1))); do
  CT=mqns_srt_$I
  if [[ $(docker wait $CT) -ne 0 ]]; then
    echo $CT container failed
    exit 1
  fi
  docker rm $CT
done

python srt_plot.py \
  --params "$OUTDIR"/params.toml --indir "$OUTDIR" $PLOT_FLAG \
  --csv "$OUTDIR/srt.csv" --plt "$OUTDIR/srt.png"
