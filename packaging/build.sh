#!/usr/bin/env bash
set -euo pipefail

make clean
make CC="${CC}" CFLAGS="${CFLAGS} -std=c99 -Wall -O2 -fPIE" \
  CPPFLAGS="${CPPFLAGS:-}" LDFLAGS="${LDFLAGS:-}" -j"${CPU_COUNT:-2}"
mkdir -p "${PREFIX}/bin" "${PREFIX}/share/man/man1"
install -m 0755 protlocus protlocus-recheck "${PREFIX}/bin/"
for tool in tools/*.py; do
  case "$tool" in */__init__.py|*/prepare_release.py|*/assemble_release.py|*/verify_bundle.py|*/local_conda_recipes.py|*/verify_conda_install.py) continue;; esac
  install -m 0755 "$tool" "${PREFIX}/bin/"
done
install -m 0755 tools/protlocus_hmm_pipeline.sh "${PREFIX}/bin/"
install -m 0644 protlocus.1 "${PREFIX}/share/man/man1/"

install -m 0755 tools/protlocus-example "${PREFIX}/bin/"
mkdir -p "${PREFIX}/share/protlocus"
cp -R examples "${PREFIX}/share/protlocus/examples"
