#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"
P3D_DIR="${THIRD_PARTY_DIR}/pytorch3d"
P3D_REPO="https://github.com/facebookresearch/pytorch3d.git"
# tag v0.7.5 (no prebuilt wheel exists for py311 + cu118 + torch 2.1.0)
P3D_COMMIT="2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ ! -d "${P3D_DIR}/.git" ]]; then
    git clone "${P3D_REPO}" "${P3D_DIR}"
fi

git -C "${P3D_DIR}" fetch origin "${P3D_COMMIT}" || git -C "${P3D_DIR}" fetch origin
git -C "${P3D_DIR}" checkout "${P3D_COMMIT}"

printf 'pytorch3d source ready at %s\n' "${P3D_DIR}"
