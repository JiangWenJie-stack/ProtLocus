"""Exercise complete-bundle contents, checksums, and unsafe archive rejection."""
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.assemble_release import assemble, extract


class BundleAssemblyTest(unittest.TestCase):
    def archive(self, path, files):
        with tarfile.open(path, 'w:gz') as archive:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

    def test_complete_bundle_has_all_backends_and_full_source(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            version = '0.1.0'
            names = {'cpu': 'protlocus', 'opencl': 'protlocus-gpu', 'cuda': 'protlocus-cuda'}
            for backend, command in names.items():
                stem = 'protlocus-' + version + '-linux-x86_64-' + backend
                self.archive(assets / (stem + '.tar.gz'), {
                    stem + '/VERSION': b'0.1.0\n',
                    stem + '/bin/' + command: backend.encode(),
                    stem + '/docs/original.md': b'earlier docs',
                })
            self.archive(assets / 'protlocus-0.1.0.tar.gz', {
                'protlocus-0.1.0/VERSION': b'0.1.0\n',
                'protlocus-0.1.0/source.c': b'full source',
                'protlocus-0.1.0/docs/current.md': b'current docs',
                'protlocus-0.1.0/packaging/BINARY_README.md': b'English usage',
                'protlocus-0.1.0/tools/verify_bundle.py': b'# verification',
            })
            channel = assets / 'conda-channel/linux-64'
            channel.mkdir(parents=True)
            (channel / 'example.conda').write_bytes(b'installable package fixture')
            bundle = assemble(assets, version)
            target = assets / 'extracted'
            extract(bundle, target)
            root = target / 'protlocus-0.1.0-linux-x86_64-full'
            self.assertEqual((root / 'conda-channel/linux-64/example.conda').read_bytes(),
                             b'installable package fixture')
            for command in names.values():
                self.assertTrue((root / 'bin' / command).is_file())
            self.assertEqual((root / 'src/source.c').read_bytes(), b'full source')
            self.assertTrue((root / 'docs/current.md').is_file())
            for row in (root / 'SHA256SUMS').read_text().splitlines():
                digest, name = row.split('  ', 1)
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                assemble(assets, version)

    def test_path_traversal_in_input_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'bad.tar.gz'
            self.archive(archive, {'../outside': b'invalid'})
            with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
                extract(archive, root / 'extract')
            self.assertFalse((root / 'outside').exists())


if __name__ == '__main__':
    unittest.main()
