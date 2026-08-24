#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
P3D_DIR="${PROJECT_ROOT}/third_party/pytorch3d"
CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-}}"
export PYTHONNOUSERSITE=1

# Same cross-environment flag hygiene as build_minkowski.sh / build_cropformer_ops.sh.
unset CFLAGS CXXFLAGS LDFLAGS TORCH_CUDA_ARCH_LIST

if [[ ! -d "${P3D_DIR}" ]]; then
    printf 'pytorch3d source not found: %s\n' "${P3D_DIR}" >&2
    printf 'Run bootstrap_pytorch3d first.\n' >&2
    exit 1
fi

if [[ -z "${CUDA_HOME}" || ! -d "${CUDA_HOME}" ]]; then
    printf 'CUDA_HOME is not set and could not be inferred from CONDA_PREFIX.\n' >&2
    exit 1
fi

export CUDA_HOME
export FORCE_CUDA=1
# nvcc jobs eat several GB each; nproc parallelism OOMs a 32 GB host.
DEFAULT_JOBS=$(( $(nproc) < 4 ? $(nproc) : 4 ))
export MAX_JOBS="${MAX_JOBS:-${DEFAULT_JOBS}}"

python "${SCRIPT_DIR}/patch_pytorch3d_sources.py" "${P3D_DIR}"

# --no-deps: runtime deps (fvcore, iopath, torch) are pinned by the pixi env.
pip install --no-deps --no-build-isolation "${P3D_DIR}"

python -c "import pytorch3d; from pytorch3d.ops import ball_query; print('pytorch3d', pytorch3d.__version__)"

mkdir -p "${PROJECT_ROOT}/.build"
touch "${PROJECT_ROOT}/.build/pytorch3d.ok"
printf 'pytorch3d built and importable.\n'
