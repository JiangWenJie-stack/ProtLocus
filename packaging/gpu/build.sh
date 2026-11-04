#!/usr/bin/env bash
set -euo pipefail
make clean
make gpu=1 CC="${CC}" CFLAGS="${CFLAGS} -std=c99 -Wall -O2 -fPIE" \
  CPPFLAGS="${CPPFLAGS:-}" LDFLAGS="${LDFLAGS:-}" -j"${CPU_COUNT:-2}"
"${READELF:-${HOST}-readelf}" -d protlocus | grep -F "libOpenCL.so"
mkdir -p "${PREFIX}/bin"
install -m 0755 protlocus "${PREFIX}/bin/protlocus-gpu"
