"""Validate complete offline references and recover their annotated gene loci."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OfflineExamplesTest(unittest.TestCase):
    def test_complete_reference_sequences_fit_in_ten_mb(self):
        expected = {'mouse-mt': ('NC_005089.1', 16299), 'ecoli': ('NC_000913.3', 4641652)}
        total = 0
        for name, (accession, length) in expected.items():
            folder = ROOT / 'examples' / name
            meta = json.loads((folder / 'metadata.json').read_text())
            lines = (folder / 'genome.fa').read_text().splitlines()
            self.assertEqual(sum(line.startswith('>') for line in lines), 1)
            self.assertEqual(lines[0].split()[0], '>' + accession)
            seq = ''.join(lines[1:])
            self.assertEqual(len(seq), length)
            self.assertEqual(hashlib.sha256(seq.encode()).hexdigest(), meta['sequence_sha256'])
            total += sum(f.stat().st_size for f in folder.iterdir() if f.is_file())
        self.assertLess(total, 10_000_000)

    def test_both_examples_recover_the_annotated_locus(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('mouse-mt', 'ecoli'):
                with self.subTest(example=name):
                    output = Path(tmp) / name
                    result = subprocess.run([sys.executable, str(ROOT / 'tools/protlocus-example'), name,
                                             '--binary', str(ROOT / 'protlocus'), '--outdir', str(output)],
                                            check=True, text=True, capture_output=True)
                    self.assertIn('PASS ' + name, result.stdout)
                    report = json.loads((output / 'report.json').read_text())
                    self.assertTrue(report['passed'])
                    self.assertEqual(report['hits'][0]['query_coverage'], 1.0)
                    self.assertTrue((output / 'result.gff3').read_text().startswith('##gff-version 3'))
                    # A repeated command must protect the user's existing results.
                    repeated = subprocess.run(result.args, capture_output=True, text=True)
                    self.assertNotEqual(repeated.returncode, 0)
                    self.assertIn('Output directory must be empty', repeated.stderr)
