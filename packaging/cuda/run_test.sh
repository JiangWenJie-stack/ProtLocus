#!/usr/bin/env bash
set -euo pipefail
# Conda builders have no GPU. This checks linkage and the CPU fallback only.
# A real device test is a separate workflow and must reject GPU fallback.
test "$(protlocus-cuda --version)" = "$(protlocus --version)"
ldd "$(command -v protlocus-cuda)" > linkage.txt
grep -F "libcudart.so" linkage.txt
if grep -q "not found" linkage.txt; then cat linkage.txt >&2; exit 1; fi
protlocus-cuda test/DPP3-hs.gen.fa.gz test/DPP3-mm.pep.fa.gz > example.paf
python3 - <<'CHECK'
import hashlib
from pathlib import Path
assert hashlib.sha256(Path('example.paf').read_bytes()).hexdigest() == 'd303e5f39bf5c0002421bbc7d8cc3adf7827ef2610e32619b226d556cf343d04'
CHECK
