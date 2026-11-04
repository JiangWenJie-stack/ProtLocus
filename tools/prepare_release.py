#!/usr/bin/env python3
"""Create a tracked-source archive and matching CPU/OpenCL/CUDA Bioconda recipes, without uploading."""
import argparse
import gzip
import hashlib
import io
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def check_versions(root=ROOT):
    version = (root / 'VERSION').read_text().strip()
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('VERSION must contain a numbered release, for example 0.1.0')
    markers = {
        'protlocus.h': r'#define PROTLOCUS_VERSION "([^"]+)"',
        'recheck.c': r'#define RC_VERSION "([^"]+)"',
        'tools/protlocus_genemodel.py': r"^VERSION = '([^']+)'",
    }
    for name, pattern in markers.items():
        match = re.search(pattern, (root / name).read_text(), re.MULTILINE)
        if not match or match[1] != version:
            raise ValueError('Version does not match VERSION: ' + name)
    return version


def package_release(repository, maintainer, output, tag=None, root=ROOT):
    # These inputs become URLs and YAML values, so reject control characters and
    # punctuation rather than treating repository metadata as executable text.
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+', repository):
        raise ValueError('repository must be OWNER/REPO')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}', maintainer):
        raise ValueError('maintainer must be an individual GitHub username')
    version = check_versions(root)
    if tag is not None:
        if tag != 'v' + version:
            raise ValueError('Release tag must be v' + version)
        if git('rev-parse', tag + '^{commit}', root=root) != git('rev-parse', 'HEAD', root=root):
            raise ValueError('Release tag must point to the checked-out commit')
    if subprocess.call(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--']):
        raise ValueError('Commit tracked changes before preparing a release')
    output = output.resolve()
    names = ['protlocus-' + version + '.tar.gz', 'bioconda-recipe-' + version + '.tar.gz', 'SHA256SUMS']
    if any((output / name).exists() for name in names) or (output / 'bioconda-recipe').exists():
        raise ValueError('Release output already exists; choose a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    source_tar = git('archive', '--format=tar', '--prefix=protlocus-' + version + '/', 'HEAD', root=root)
    source = output / names[0]
    source.write_bytes(gzip.compress(source_tar, mtime=0))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    recipe_root = output / 'bioconda-recipe'
    for variant, package in [('', 'protlocus'), ('gpu', 'protlocus-gpu'), ('cuda', 'protlocus-cuda')]:
        recipe = recipe_root / 'recipes' / package
        recipe.mkdir(parents=True)
        template_dir = root / 'packaging' / variant
        text = (template_dir / 'meta.yaml.template').read_text()
        for key, value in {'VERSION': version, 'REPOSITORY': repository, 'SHA256': digest, 'MAINTAINER': maintainer}.items():
            text = text.replace('__' + key + '__', value)
        if re.search(r'__[A-Z]+__', text):
            raise ValueError('Unresolved recipe placeholder')
        (recipe / 'meta.yaml').write_text(text)
        for name in ('build.sh', 'run_test.sh', 'conda_build_config.yaml'):
            if (template_dir / name).exists():
                shutil.copy2(template_dir / name, recipe / name)
    recipe_bytes = io.BytesIO()
    with tarfile.open(fileobj=recipe_bytes, mode='w') as archive:
        for file in sorted(recipe_root.rglob("*")):
            if not file.is_file():
                continue
            info = tarfile.TarInfo(file.relative_to(recipe_root).as_posix())
            info.size = file.stat().st_size
            info.mode = 0o755 if file.suffix == '.sh' else 0o644
            archive.addfile(info, io.BytesIO(file.read_bytes()))
    recipe_tar = output / names[1]
    recipe_tar.write_bytes(gzip.compress(recipe_bytes.getvalue(), mtime=0))
    (output / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256((output / name).read_bytes()).hexdigest() + '  ' + name + '\n'
        for name in names[:2]
    ))
    return source, recipe_tar, output / 'SHA256SUMS'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', help='Actual GitHub OWNER/REPO')
    parser.add_argument('--maintainer', help='Individual GitHub username for recipe maintenance')
    parser.add_argument('--tag', help='Validate a GitHub Release tag, e.g. v0.1.0')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    parser.add_argument('--check', action='store_true', help='Only check version consistency; no Git repository required')
    args = parser.parse_args()
    try:
        if args.check:
            print(check_versions())
            return
        if not args.repository or not args.maintainer:
            parser.error('--repository and --maintainer are required for packaging')
        for artifact in package_release(args.repository, args.maintainer, args.output, args.tag):
            print(artifact)
    except (ValueError, subprocess.CalledProcessError, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
