#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ "$#" -gt 0 ]]; then
  scales=("$@")
else
  scales=(0.8 1.0 1.2)
fi

cache_prefix="robots/leap_hand/caches/ball_grasp_hora_sharpa_style_50k"

for scale in "${scales[@]}"; do
  # Match resolve_grasp_cache_file()'s canonical ``:g`` tag. In particular,
  # the 1.0 bucket is stored as ``_1.npy`` rather than ``_1.0.npy``.
  scale_tag="$(printf '%g' "${scale}")"
  output_path="${cache_prefix}_${scale_tag}.npy"
  echo "Collecting LEAP HORA grasp cache: scale=${scale}, output=${output_path}"
  uv run train \
    --algo appo \
    --task leap_inhand_ball_grasp_allegro \
    --sim mujoco \
    training.no_play=true \
    env.object_scale="${scale}" \
    env.grasp_cache_path="${output_path}"
done
