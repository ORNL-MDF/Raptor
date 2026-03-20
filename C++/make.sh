#!/bin/bash
set -euo pipefail

# Anchor the C++ build to this script's directory so callers do not depend on cwd.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${REPO_ROOT}/build"

# Allow incremental builds by default while preserving an easy clean-build switch.
if [[ "${CLEAN_BUILD:-0}" == "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi

export NVCC_WRAPPER_DEFAULT_COMPILER=mpic++
export MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi

mkdir -p "${BUILD_DIR}"
pushd "${BUILD_DIR}" >/dev/null

cmake \
  -D CMAKE_BUILD_TYPE="Release" \
  -D CMAKE_INSTALL_PREFIX=install \
  -D CMAKE_CXX_FLAGS="-fopenmp -O3 -ffast-math -march=znver3 -mtune=znver3" \
  -D CMAKE_PREFIX_PATH="${KOKKOS_DIR};${MPI_DIR}" \
  -D MPI_CXX_COMPILER=/usr/bin/mpicxx \
  -D CMAKE_CUDA_ARCHITECTURES="86" \
  "${REPO_ROOT}"

make -j 8 install

popd >/dev/null
