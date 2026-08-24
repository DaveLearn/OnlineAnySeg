#!/usr/bin/env bash
# Unpack the vendored GT segmentation archives used by the native evaluation
# (eval/evaluate_seqs.py). Sentinel files make this an outputs-cached pixi task.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

unzip -o -q "${PROJECT_ROOT}/eval/scannet200/validation/scannet_gt_seg.zip" -d "${PROJECT_ROOT}/eval/scannet200/validation/"
unzip -o -q "${PROJECT_ROOT}/eval/sceneNN/sceneNN_gt_seg.zip" -d "${PROJECT_ROOT}/eval/sceneNN/"

touch "${PROJECT_ROOT}/eval/.gt_seg_unpacked"
printf 'Native eval GT archives unpacked.\n'
