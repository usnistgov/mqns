#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

info() {
  echo -ne "\e[0;32m"
  echo -n "$*"
  echo -e "\e[0m"
}

die() {
  echo -ne "\e[0;31m"
  echo -n "$*"
  echo -e "\e[0m"
  exit 1
}

PY_ARGS=()
RS_ARGS=()
ARGS_SPLIT=0
for A in "$@"; do
  if [[ "$A" == '--' ]]; then
    ARGS_SPLIT=$((ARGS_SPLIT+1))
    continue
  fi
  if [[ "$A" == '-h' || "$A" == '--help' ]]; then
    info 'Usage: bash demo.sh [COMMON-ARGS] -- [PY-ARGS] -- [RS-ARGS]'
    info '  COMMON-ARGS: passed to both Python and Rust'
    info '  PY-ARGS: passed to Python script'
    info '  RS-ARGS: passed to Rust crate'
    info '----------------------------------------------------------------'
    info Python flags:
    python extctrl_dp.py --help
    info '----------------------------------------------------------------'
    info Rust flags:
    cargo run -q -- --help
    exit 0
  fi
  if [[ $ARGS_SPLIT -eq 0 ]]; then
    PY_ARGS+=("$A")
    RS_ARGS+=("$A")
  elif [[ $ARGS_SPLIT -eq 1 ]]; then
    PY_ARGS+=("$A")
  elif [[ $ARGS_SPLIT -eq 2 ]]; then
    RS_ARGS+=("$A")
  fi
done

NATS_PREFIX=${MQNS_NATS_PREFIX:-mqns.classicbridge}
STREAM=${MQNS_NATS_STREAM:-MQNS_CLASSIC_BRIDGE}

if ! command -v nats &>/dev/null; then
  die NATS client not installed, please install from https://github.com/nats-io/natscli
fi
if ! nats account info &>/dev/null; then
  die NATS server not running, please start NATS server
fi

info Deleting existing NATS stream if exists
nats stream rm $STREAM -f || true

info Defining NATS stream
nats stream add $STREAM \
  --subjects "$NATS_PREFIX.*.*" \
  --storage memory \
  --retention limits \
  --discard old \
  --max-msgs=-1 \
  --max-bytes=-1 \
  --max-age=1h \
  --defaults

trap 'info Stopping; kill $(jobs -p) 2>/dev/null || true' EXIT

info Launching data plane '(Python)'
python extctrl_dp.py --nats_prefix=$NATS_PREFIX "${PY_ARGS[@]}" &

info Launching control plane '(Rust)'
cargo run -- --nats_prefix=$NATS_PREFIX "${RS_ARGS[@]}"

info Control plane stopped, giving the data plane a chance to finish and print statistics
sleep 1
