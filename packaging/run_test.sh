#!/usr/bin/env bash
set -euo pipefail
version=$(protlocus --version)
test "$(protlocus-recheck -v)" = "$version"
test "$(python3 "$PREFIX/bin/protlocus_genemodel.py" --version)" = "$version"
protlocus test/DPP3-hs.gen.fa.gz test/DPP3-mm.pep.fa.gz > example.paf
python3 - <<'PY'
import hashlib
from pathlib import Path
expected = 'd303e5f39bf5c0002421bbc7d8cc3adf7827ef2610e32619b226d556cf343d04'
actual = hashlib.sha256(Path('example.paf').read_bytes()).hexdigest()
assert actual == expected, actual
PY

protlocus-example mouse-mt --outdir example-mouse-mt
protlocus-example ecoli --outdir example-ecoli
