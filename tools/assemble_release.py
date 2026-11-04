#!/usr/bin/env python3
"""Combine source and three Linux binary archives into a complete release bundle."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile


def extract(archive, directory):
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member: ' + member.name)
        tar.extractall(directory)


def assemble(assets, version):
    if not version or any(c not in '0123456789.' for c in version):
        raise ValueError('Invalid version')
    name = 'protlocus-' + version + '-linux-x86_64-full'
    output = assets / (name + '.tar.gz')
    if output.exists():
        raise ValueError('Refusing to overwrite an existing complete release')
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        full = temp / name
        full.mkdir()
        for backend in ('cpu', 'opencl', 'cuda'):
            stem = 'protlocus-' + version + '-linux-x86_64-' + backend
            extract(assets / (stem + '.tar.gz'), temp / backend)
            root = temp / backend / stem
            if (root / 'VERSION').read_text().strip() != version:
                raise ValueError('Binary/source version mismatch: ' + backend)
            for file in sorted(root.rglob('*')):
                if file.is_file() and file.name != 'SHA256SUMS':
                    target = full / file.relative_to(root)
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, target)
        source_name = 'protlocus-' + version
        extract(assets / (source_name + '.tar.gz'), temp / 'source')
        shutil.copytree(temp / 'source' / source_name, full / 'src')
        # Include real installable Conda packages when supplied by the release job.
        channel = assets / 'conda-channel'
        if channel.is_dir():
            shutil.copytree(channel, full / 'conda-channel',
                            ignore=shutil.ignore_patterns('.cache', '__pycache__'))
        # Use the final source tree's documentation in the complete bundle.
        shutil.rmtree(full / 'docs')
        shutil.copytree(full / 'src/docs', full / 'docs')
        shutil.copy2(full / 'src/packaging/BINARY_README.md', full / 'README.md')
        shutil.copy2(full / 'src/tools/verify_bundle.py', full / 'verify.py')
        (full / 'SHA256SUMS').write_text(''.join(
            hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + f.relative_to(full).as_posix() + '\n'
            for f in sorted(full.rglob('*')) if f.is_file() and f != full / 'SHA256SUMS'))
        with tarfile.open(output, 'w:gz') as tar:
            tar.add(full, arcname=name)
    (assets / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + f.name + '\n'
        for f in sorted(assets.glob('*.tar.gz'))))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    print(assemble(args.assets, args.version))


if __name__ == '__main__':
    main()
