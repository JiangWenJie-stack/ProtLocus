#!/usr/bin/env python3

import io
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_accuracy import evaluate, read_predictions, read_truth
from tools.hmmrecheck import dedup_peptides, parse_domtbl, read_candidates
from tools.junctions_to_spsc import convert_bed6, convert_star, deduplicate
from tools import protlocus_genemodel as gm


class JunctionConversionTest(unittest.TestCase):
    def test_star_strands_and_coordinates(self) -> None:
        source = io.StringIO(
            "chr1\t101\t200\t1\t1\t1\t7\t4\t30\n"
            "chr1\t301\t400\t2\t1\t0\t3\t0\t30\n"
        )
        records = deduplicate(convert_star(source))
        self.assertIn(("chr1", 100, "+", "D", 8), records)
        self.assertIn(("chr1", 200, "+", "A", 8), records)
        self.assertIn(("chr1", 400, "-", "D", 4), records)
        self.assertIn(("chr1", 300, "-", "A", 4), records)

    def test_bed_coordinates(self) -> None:
        records = deduplicate(convert_bed6(io.StringIO("chr2\t10\t20\tj\t8\t-\n")))
        self.assertEqual(records, [("chr2", 10, "-", "A", 8), ("chr2", 20, "-", "D", 8)])


class AccuracyEvaluationTest(unittest.TestCase):
    def test_exact_model(self) -> None:
        truth_text = "query\tcontig\tstrand\tcds\nq1\tchr1\t+\t101-150,201-250\n"
        prediction_text = (
            "chr1\tProtLocus\tmRNA\t101\t250\t100\t+\t.\tID=m1;Rank=1;Target=q1 1 34\n"
            "chr1\tProtLocus\tCDS\t101\t150\t50\t+\t0\tParent=m1;Rank=1\n"
            "chr1\tProtLocus\tCDS\t201\t250\t50\t+\t0\tParent=m1;Rank=1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            truth_path = Path(directory) / "truth.tsv"
            prediction_path = Path(directory) / "prediction.gff"
            truth_path.write_text(truth_text, encoding="utf-8")
            prediction_path.write_text(prediction_text, encoding="utf-8")
            result = evaluate(read_truth(truth_path), read_predictions(prediction_path), 0, 0.1)
        self.assertEqual(result["top1_locus_recall"], 1.0)
        self.assertEqual(result["exact_cds_model_recall"], 1.0)
        self.assertEqual(result["junction_precision"], 1.0)
        self.assertEqual(result["junction_recall"], 1.0)


class HmmRecheckParsingTest(unittest.TestCase):
    PAF = "PF00001\t100\t0\t90\t+\tchr1\t50000\t1000\t2000\t80\t270\t60\tAS:i:123\tps:i:88"

    def test_read_candidates_pairs_sta(self) -> None:
        lines = [
            self.PAF + "\n",
            "##STA\tMKT*LLV\n",
            "##ATA\tignored\n",
            "unmapped\t10\t0\t0\t*\t*\t0\t0\t0\t0\t0\t0\n",
            "##PAF\t" + self.PAF + "\n",      # GFF 模式前缀
            "##STA\tAAAA\n",
            "chr1\tProtLocus\tmRNA\t1\t9\t1\t+\t.\tID=x\n",  # GFF 行应跳过
        ]
        cands = read_candidates(lines)
        self.assertEqual(len(cands), 2)
        self.assertEqual(cands[0]["pep"], "MKT*LLV")
        self.assertEqual(cands[0]["family"], "PF00001")
        self.assertEqual((cands[0]["AS"], cands[0]["ps"]), (123, 88))
        self.assertEqual((cands[0]["gstart"], cands[0]["gend"], cands[0]["strand"]),
                         (1000, 2000, "+"))
        self.assertEqual(cands[1]["pep"], "AAAA")

    def test_dedup_and_stop_replacement(self) -> None:
        lines = [self.PAF + "\n", "##STA\tMKT*LLV\n",
                 self.PAF + "\n", "##STA\tMKT*LLV\n",
                 self.PAF + "\n"]  # 第三条无 ##STA
        cands = read_candidates(lines)
        families, pid2cands, n_skip = dedup_peptides(cands, keep_stop=False)
        self.assertEqual(n_skip, 1)
        self.assertEqual(len(families["PF00001"]), 1)
        pid, pep = families["PF00001"][0]
        self.assertEqual(pep, "MKTXLLV")           # '*' -> 'X'
        self.assertEqual(pid2cands[pid], [0, 1])   # 两条候选共享一个肽段
        self.assertEqual(cands[0]["n_stop"], 1)

    def test_parse_domtbl_best_domain(self) -> None:
        row = ("p0 - 7 PF00001 - 90 {fullE} 55.0 0.1 1 2 1e-3 {iE} {sc} 0.5"
               " 1 80 2 85 1 90 0.9 -")
        lines = ["# comment",
                 row.format(fullE="1e-10", iE="1e-8", sc="40.0"),
                 row.format(fullE="1e-10", iE="1e-12", sc="55.0")]
        best = parse_domtbl(lines)
        self.assertEqual(best["p0"], (1e-12, 55.0, 1e-10))


class GeneModelTest(unittest.TestCase):
    def test_cigar_blocks_phases(self) -> None:
        # 10 个密码子, 0 相位内含子 (N) 与 1 相位内含子 (U: 内含子 + 3nt 密码子)
        self.assertEqual(gm.cigar_blocks("4M100N6M", 1000),
                         [(1000, 1012, 1000 % 3), (1112, 1130, (1112 + 0) % 3)])
        blocks = gm.cigar_blocks("4M103U5M", 0)
        self.assertEqual(blocks, [(0, 13, 0), (113, 130, (113 + 2) % 3)])
        self.assertIsNone(gm.cigar_blocks("4M1F5M", 0))

    def test_direct_fusion_and_orf_ends(self) -> None:
        import argparse
        # ATG + 18 个无终止子密码子 + TAA, 其中间一段被两个 "证据单元" 覆盖
        orf = "ATG" + "GCT" * 8 + "AAA" * 8 + "GAA" * 8 + "TAA"
        seq = "CCCC" + orf + "CCCC"
        M = gm.default_model()
        P = argparse.Namespace(start_slack=8.0, max_ext=100, max_ext_introns=0, max_introns=0,
                               max_src_atg=50, nostart_penalty=8.0, ext_min=30.0, cap_exon=40.0,
                               exon_cost=6.0, intron_cost=4.0, min_exon=12, max_mid_exon=500,
                               min_donor=0.0, min_acceptor=0.0, link_min=8.0, max_link=1000,
                               sep_bonus=0.0, min_aa=5)
        a, b = 4 + 3 + 24, 4 + 3 + 24 + 24 + 12      # 两个同框证据单元, 中间无终止子
        g = (4 + 3) % 3
        u1, u2 = gm.Unit(), gm.Unit()
        u1.blocks, u1.support = [(4 + 3, a, g)], [("F1", 1e-9, 50.0)]
        u2.blocks, u2.support = [(b, b + 12, g)], [("F2", 1e-9, 50.0)]
        u1.span(), u2.span()
        chains = gm.assemble_chains(seq, [u1, u2], M, P)
        self.assertEqual(len(chains), 1)
        genes = gm.make_genes("c", seq, "+", chains, M, P)
        self.assertEqual(len(genes), 1)
        self.assertEqual(genes[0]["blocks"], [(4, 4 + len(orf))])
        self.assertEqual(genes[0]["prot"], "M" + "A" * 8 + "K" * 8 + "E" * 8 + "*")
        self.assertFalse(genes[0]["p5"] or genes[0]["p3"])


if __name__ == "__main__":
    unittest.main()