"""Check release provenance and failure behavior without contacting GitHub."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

from tools.prepare_release import check_versions, package_release

ROOT = Path(__file__).resolve().parents[1]


class ReleasePackagingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'
        self.root.mkdir()
        for name in ('VERSION', 'protlocus.h', 'recheck.c', 'LICENSE.txt',
                     'tools/protlocus_genemodel.py', 'packaging/meta.yaml.template',
                     'packaging/build.sh', 'packaging/run_test.sh',
                     'packaging/conda_build_config.yaml'):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        for variant in ('gpu', 'cuda'):
            shutil.copytree(ROOT / 'packaging' / variant, self.root / 'packaging' / variant)
        self.run_git('init', '-q')
        self.run_git('add', '.')
        self.run_git('-c', 'user.name=Local packaging test', '-c',
                     'user.email=packaging-test@example.invalid', 'commit', '-qm', 'Fixture')
        self.version = check_versions(self.root)
        self.run_git('tag', 'v' + self.version)
        self.output = Path(self.temp.name) / 'output'

    def run_git(self, *args):
        subprocess.run(['git', '-C', str(self.root), *args], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_archive_and_recipe_refer_to_the_same_source(self):
        (self.root / 'untracked-secret.txt').write_text('must never be archived')
        source, recipe, sums = package_release('example/ProtLocus', 'example', self.output,
                                               'v' + self.version, root=self.root)
        with tarfile.open(source) as archive:
            names = archive.getnames()
            self.assertFalse(any('secret' in name for name in names))
            license_file = archive.extractfile('protlocus-' + self.version + '/LICENSE.txt')
            self.assertEqual(license_file.read(), (ROOT / 'LICENSE.txt').read_bytes())
        with tarfile.open(recipe) as archive:
            meta = archive.extractfile('recipes/protlocus/meta.yaml').read().decode()
            self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), meta)
            self.assertIn('/example/ProtLocus/releases/download/v{{ version }}/', meta)
            self.assertIn('    - example\n', meta)
            self.assertNotIn('__SHA256__', meta)
            self.assertIn('recipes/protlocus/run_test.sh', archive.getnames())
            self.assertIn('recipes/protlocus-gpu/meta.yaml', archive.getnames())
            for name in ('protlocus', 'protlocus-gpu', 'protlocus-cuda'):
                config = archive.extractfile('recipes/' + name + '/conda_build_config.yaml').read().decode()
                self.assertIn("c_stdlib_version:\n  - '2.28'", config)
            for package in ('protlocus-gpu', 'protlocus-cuda'):
                text = archive.extractfile('recipes/' + package + '/meta.yaml').read().decode()
                self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), text)
        self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), sums.read_text())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            package_release('example/ProtLocus', 'example', self.output, root=self.root)

    def test_conda_cppflags_preserve_gpu_backend_selection(self):
        for arguments, flag in [(['gpu=1'], '-DNS_OPENCL'),
                                (['cuda=1', 'CUDA_HOME=/test/cuda'], '-DNS_CUDA')]:
            result = subprocess.run(['make', '-Bn', *arguments, 'CPPFLAGS=-DCONDA_TEST',
                                     'kalloc.o'], cwd=ROOT, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.assertIn('-DCONDA_TEST', result.stdout)
            self.assertIn(flag, result.stdout)

    def test_wrong_tag_is_rejected_before_creating_assets(self):
        with self.assertRaisesRegex(ValueError, 'tag must be'):
            package_release('example/ProtLocus', 'example', self.output, 'v9.9.9', root=self.root)
        self.assertFalse(self.output.exists())

    def test_uncommitted_source_is_rejected(self):
        path = self.root / 'LICENSE.txt'
        path.write_text(path.read_text() + '\nUncommitted change\n')
        with self.assertRaisesRegex(ValueError, 'Commit tracked changes'):
            package_release('example/ProtLocus', 'example', self.output, root=self.root)
        self.assertFalse(self.output.exists())

    def test_inconsistent_version_is_rejected(self):
        (self.root / 'VERSION').write_text('9.9.9\n')
        with self.assertRaisesRegex(ValueError, 'does not match VERSION'):
            check_versions(self.root)


if __name__ == '__main__':
    unittest.main()
