#!/usr/bin/env python3
"""Verify installed commands and shared-library linkage, optionally requiring a GPU."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 'd303e5f39bf5c0002421bbc7d8cc3adf7827ef2610e32619b226d556cf343d04'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'opencl', 'cuda', 'all'], default='all')
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()
    programs = {'cpu': ('protlocus', None), 'opencl': ('protlocus-gpu', 'libOpenCL.so'),
                'cuda': ('protlocus-cuda', 'libcudart.so')}
    expected_version = (ROOT / 'VERSION').read_text().strip()
    for backend, (command, library) in programs.items():
        if args.backend not in ('all', backend):
            continue
        binary = shutil.which(command)
        if not binary:
            raise RuntimeError('Command not installed: ' + command)
        version = subprocess.check_output([binary, '--version'], text=True).strip()
        if version != expected_version:
            raise RuntimeError(command + ': unexpected version ' + version)
        linkage = subprocess.check_output(['ldd', binary], text=True)
        if 'not found' in linkage or (library and library not in linkage):
            raise RuntimeError(command + ': wrong/missing runtime library\n' + linkage)
        options = ['--gpu', '--gpu-min-cells=0'] if args.require_gpu and library else []
        result = subprocess.run([binary, *options, str(ROOT / 'test/DPP3-hs.gen.fa.gz'),
                                 str(ROOT / 'test/DPP3-mm.pep.fa.gz')], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if hashlib.sha256(result.stdout).hexdigest() != EXPECTED:
            raise RuntimeError(command + ': example output mismatch')
        if options and ('[M::gpu] using device' not in result.stderr.decode() or
                        'GPU unavailable' in result.stderr.decode()):
            raise RuntimeError(command + ': a real GPU was required but was not used')
        print('PASS:', command, 'version, linkage, and example;' ,
              'GPU initialized' if options else 'default CPU execution')
    recheck = subprocess.check_output(['protlocus-recheck', '-v'], text=True).strip()
    if recheck != expected_version:
        raise RuntimeError('Rechecker version mismatch')
    subprocess.run(['protlocus_hmm_pipeline.sh', '-h'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['protlocus_genemodel.py', '--help'], check=True, stdout=subprocess.DEVNULL)
    print('PASS: shared rechecker and companion tool entry points')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
