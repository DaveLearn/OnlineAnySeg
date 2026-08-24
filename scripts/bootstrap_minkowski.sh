#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"
ME_DIR="${THIRD_PARTY_DIR}/MinkowskiEngine"
ME_REPO="https://github.com/NVIDIA/MinkowskiEngine.git"
# v0.5.4 era master HEAD (last upstream commit, 2023-08); ME is unmaintained.
ME_COMMIT="02fc608bea4c0549b0a7b00ca1bf15dee4a0b228"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ ! -d "${ME_DIR}/.git" ]]; then
    git clone "${ME_REPO}" "${ME_DIR}"
fi

git -C "${ME_DIR}" fetch origin "${ME_COMMIT}" || git -C "${ME_DIR}" fetch origin
git -C "${ME_DIR}" checkout "${ME_COMMIT}"

printf 'MinkowskiEngine ready at %s\n' "${ME_DIR}"
