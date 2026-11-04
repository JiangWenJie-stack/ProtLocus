#!/usr/bin/env bash
set -euo pipefail
make clean
# A fat binary for Volta through Hopper, plus PTX for forward compatibility.
GENCODE="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_80,code=compute_80"
make cuda=1 CC="${CC}" NVCC="${CUDACXX:-nvcc}" CUDA_HOME="${PREFIX}" \
  CUDA_GENCODE="$GENCODE" NVCCFLAGS="-ccbin ${CXX}" \
  CFLAGS="${CFLAGS} -std=c99 -Wall -O2 -fPIE" CPPFLAGS="${CPPFLAGS:-}" \
  LDFLAGS="${LDFLAGS:-}" -j"${CPU_COUNT:-2}"
"${READELF:-${HOST}-readelf}" -d protlocus | grep -F "libcudart.so"
mkdir -p "${PREFIX}/bin"
install -m 0755 protlocus "${PREFIX}/bin/protlocus-cuda"
