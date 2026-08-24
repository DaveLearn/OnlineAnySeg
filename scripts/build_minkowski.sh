#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ME_DIR="${PROJECT_ROOT}/third_party/MinkowskiEngine"
CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-}}"
export PYTHONNOUSERSITE=1

# Same cross-environment flag hygiene as build_cropformer_ops.sh: a parent pixi
# shell's cuda-toolkit activation exports CFLAGS/CXXFLAGS/LDFLAGS pointing at
# its own environment, which mixes headers/libs from two environments into one
# .so. Clear them and let torch auto-detect the local GPU architecture.
unset CFLAGS CXXFLAGS LDFLAGS TORCH_CUDA_ARCH_LIST

if [[ ! -d "${ME_DIR}" ]]; then
    printf 'MinkowskiEngine source not found: %s\n' "${ME_DIR}" >&2
    printf 'Run bootstrap_minkowski first.\n' >&2
    exit 1
fi

if [[ -z "${CUDA_HOME}" || ! -d "${CUDA_HOME}" ]]; then
    printf 'CUDA_HOME is not set and could not be inferred from CONDA_PREFIX.\n' >&2
    exit 1
fi

export CUDA_HOME
# nvcc jobs eat several GB each; nproc parallelism OOMs a 32 GB host.
DEFAULT_JOBS=$(( $(nproc) < 4 ? $(nproc) : 4 ))
export MAX_JOBS="${MAX_JOBS:-${DEFAULT_JOBS}}"

python "${SCRIPT_DIR}/patch_minkowski_sources.py" "${ME_DIR}"

(cd "${ME_DIR}" && python setup.py install --blas=openblas --blas_include_dirs="${CONDA_PREFIX}/include" --force_cuda)

python -c "import MinkowskiEngine as ME; print('MinkowskiEngine', ME.__version__)"

mkdir -p "${PROJECT_ROOT}/.build"
touch "${PROJECT_ROOT}/.build/minkowski.ok"
printf 'MinkowskiEngine built and importable.\n'
