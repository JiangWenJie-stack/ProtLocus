#!/usr/bin/env python3

import hashlib
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "protlocus"
DPP3_GENOME = ROOT / "test" / "DPP3-hs.gen.fa.gz"
DPP3_PROTEIN = ROOT / "test" / "DPP3-mm.pep.fa.gz"
RESCUE_GENOME = ROOT / "test" / "rescue.gen.fa"
RESCUE_PROTEIN = ROOT / "test" / "rescue.pep.fa"
BASELINE_SHA256 = "d303e5f39bf5c0002421bbc7d8cc3adf7827ef2610e32619b226d556cf343d04"


def run_protlocus(*arguments):
    command = [str(BINARY)] + [str(argument) for argument in arguments]
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


class IntegrationTest(unittest.TestCase):
    def test_protlocus_branding(self):
        help_result = run_protlocus("--help")
        self.assertIn("Usage: protlocus ", help_result.stdout.decode())
        gff = run_protlocus("--gff", DPP3_GENOME, DPP3_PROTEIN).stdout.decode().splitlines()
        feature = next(line for line in gff if not line.startswith("#"))
        fields = feature.split("\t")
        self.assertEqual(fields[1], "ProtLocus")
        self.assertIn("ID=PL", fields[8])

    def test_default_output_is_frozen(self):
        result = run_protlocus(DPP3_GENOME, DPP3_PROTEIN)
        self.assertEqual(hashlib.sha256(result.stdout).hexdigest(), BASELINE_SHA256)

    def test_profile_does_not_change_output(self):
        baseline = run_protlocus(DPP3_GENOME, DPP3_PROTEIN)
        profiled = run_protlocus("--profile", DPP3_GENOME, DPP3_PROTEIN)
        self.assertEqual(profiled.stdout, baseline.stdout)
        profile_lines = [line for line in profiled.stderr.decode().splitlines() if line.startswith("[M::profile]")]
        self.assertEqual(len(profile_lines), 1)
        self.assertIn("query=1", profile_lines[0])
        self.assertIn("dp_candidate=1", profile_lines[0])

    def test_repeat_rescue_recovers_filtered_postings(self):
        baseline = run_protlocus("-c0", "-u", DPP3_GENOME, DPP3_PROTEIN).stdout.decode().split("\t")
        rescued = run_protlocus(
            "--accuracy",
            "--acc-no-vid-chain",
            "--acc-no-rescue",
            "--acc-no-rerank",
            "--acc-no-junc-refine",
            "-c0",
            "-u",
            DPP3_GENOME,
            DPP3_PROTEIN,
        ).stdout.decode().split("\t")
        self.assertEqual(baseline[5], "*")
        self.assertNotEqual(rescued[5], "*")
        self.assertEqual((rescued[2], rescued[3]), ("0", "738"))

    def test_anchor_fallback_and_cell_budget(self):
        common = ("-M0", "-n2", "-u", RESCUE_GENOME, RESCUE_PROTEIN)
        baseline = run_protlocus(*common).stdout.decode().split("\t")
        rescued = run_protlocus(
            "--accuracy",
            "--acc-no-repeat",
            "--acc-no-vid-chain",
            "--acc-no-rerank",
            "--acc-no-junc-refine",
            *common
        ).stdout.decode().split("\t")
        bounded = run_protlocus(
            "--accuracy",
            "--acc-no-repeat",
            "--acc-no-vid-chain",
            "--acc-no-rerank",
            "--acc-no-junc-refine",
            "--acc-rescue-cells=1",
            *common
        ).stdout.decode().split("\t")
        self.assertEqual(baseline[5], "*")
        self.assertEqual(rescued[5], "rescue_locus")
        self.assertEqual((rescued[2], rescued[3]), ("0", "32"))
        self.assertEqual(bounded[5], "*")

    def test_accuracy_output_exposes_rank(self):
        output = run_protlocus("--accuracy", DPP3_GENOME, DPP3_PROTEIN).stdout.decode()
        self.assertIn("\tra:i:", output)

    def test_junction_refinement_is_ablatable(self):
        enabled = run_protlocus("--accuracy", "--profile", DPP3_GENOME, DPP3_PROTEIN)
        disabled = run_protlocus(
            "--accuracy", "--profile", "--acc-no-junc-refine", DPP3_GENOME, DPP3_PROTEIN
        )
        budget_rejected = run_protlocus(
            "--accuracy", "--profile", "--acc-rescue-cells=19000000", DPP3_GENOME, DPP3_PROTEIN
        )
        enabled_profile = next(
            line for line in enabled.stderr.decode().splitlines() if line.startswith("[M::profile]")
        )
        disabled_profile = next(
            line for line in disabled.stderr.decode().splitlines() if line.startswith("[M::profile]")
        )
        budget_profile = next(
            line for line in budget_rejected.stderr.decode().splitlines() if line.startswith("[M::profile]")
        )
        self.assertIn("dp_realign=1 ", enabled_profile)
        self.assertIn("dp_realign_kept=0 ", enabled_profile)
        self.assertIn("dp_realign=0 ", disabled_profile)
        self.assertIn("dp_realign=0 ", budget_profile)


if __name__ == "__main__":
    unittest.main()