#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
stage="${1:-smoke}"
python_bin="${PYTHON_BIN:-./.venv/bin/python}"

case "$stage" in
  setup)
    if [[ ! -x "$python_bin" ]]; then
      python3.10 -m venv .venv
    fi
    "$python_bin" -m pip install -r requirements-sim.txt -r requirements-inference.txt
    ;;
  smoke)
    "$python_bin" scripts/check_motrixsim.py
    "$python_bin" scripts/build_scene.py
    "$python_bin" scripts/check_scene.py
    ;;
  collect)
    "$python_bin" scripts/build_scene.py
    "$python_bin" scripts/collect_color_stack.py \
      --seed "${COLLECT_SEED:-20260926}" \
      --pick-color yellow --place-color orange \
      --episodes "${EPISODES:-100}" \
      --max-attempts "${MAX_ATTEMPTS:-1000}" \
      --output "${COLLECT_OUTPUT:-data/color_stack_reproduction}"
    ;;
  eval)
    : "${CHECKPOINT_LABEL:?Set CHECKPOINT_LABEL to the checkpoint loaded by the OpenPI server}"
    "$python_bin" scripts/build_scene.py
    "$python_bin" scripts/score_robochallenge_checkpoint.py \
      --host "${OPENPI_HOST:-127.0.0.1}" --port "${OPENPI_PORT:-8000}" \
      --checkpoint "$CHECKPOINT_LABEL" \
      --seed "${EVAL_SEED:-20260928}" --episodes "${EPISODES:-20}" \
      --max-steps "${MAX_STEPS:-150}" \
      --output "${EVAL_OUTPUT:-artifacts/heldout_reproduction}" --save-video
    ;;
  *)
    echo "Usage: bash scripts/reproduce.sh {setup|smoke|collect|eval}" >&2
    exit 2
    ;;
esac
