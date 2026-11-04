#!/usr/bin/env python3
"""
把 mmseqs2 easy-search 结果 (query=六框 ORF, target=Pfam 共识序列/profile) 转成
protlocus 3.0 `--cand-table` 需要的候选区间表：

    query_name<TAB>chrom<TAB>start0<TAB>end0<TAB>strand<TAB>evalue

- query_name: Pfam 家族 NAME（去掉 hmmemit 的 "-consensus" 后缀），与 .hmm 里的 NAME 一致
- chrom/start0/end0: ORF 在基因组正链上的 0-based 半开区间 (来自 ORF fasta 头的 source=/coords=)
- strand: coords=a..b 中 a>b 为 '-'
- evalue: 该区间内所有 mmseqs 命中的最小 E-value；protlocus `--tier=fast|balanced|sensitive`
  (E<=100/1000/10000) 或 `--cand-max-evalue` 按这一列筛选，因此 mmseqs 只需按 -e 10000 跑一次。
  mmseqs tsv 没有 evalue 列时写 '.'（protlocus 视为无 E 值、总是保留）。

同一 (家族, chrom, strand) 上重叠或间距 <= --merge 的区间会合并，减少 protlocus 的 refine 次数。

用法:
  python3 mmseqs2cand.py \
      --mmseqs $OUT/yeast.mmseqs.tsv \
      --orfs   $OUT/yeast.orfs.filt.faa \
      -o       $OUT/yeast.cand.tsv
"""
import argparse
import re
import sys

COORDS_RE = re.compile(r"coords=(\d+)\.\.(\d+)")
SOURCE_RE = re.compile(r"source=(\S+)")


def parse_orf_coords(path):
    """orf_id -> (chrom, start0, end0, strand)"""
    orfs = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            orf_id = line[1:].split(None, 1)[0]
            cm = COORDS_RE.search(line)
            sm = SOURCE_RE.search(line)
            if not cm or not sm:
                continue
            a, b = int(cm.group(1)), int(cm.group(2))
            strand = "-" if a > b else "+"
            orfs[orf_id] = (sm.group(1), min(a, b) - 1, max(a, b), strand)
    return orfs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mmseqs", required=True, help="mmseqs easy-search tsv (query,target,evalue,bits)")
    ap.add_argument("--orfs", required=True, help="送入 mmseqs 的 ORF fasta (esl-translate 头, 含 source=/coords=)")
    ap.add_argument("-o", "--out", required=True, help="输出 cand-table tsv")
    ap.add_argument("--max-evalue", type=float, default=None, help="只保留 evalue <= 该值的命中 (默认不过滤)")
    ap.add_argument("--merge", type=int, default=0, help="同一(家族,chrom,strand)上间距 <= 该值(nt)的区间合并 [0: 仅合并重叠]")
    ap.add_argument("--strip-suffix", default="-consensus", help="从 target 名去掉的后缀 [-consensus]")
    args = ap.parse_args()

    orfs = parse_orf_coords(args.orfs)
    if not orfs:
        sys.exit(f"[FATAL] 未能从 {args.orfs} 解析出任何 ORF 坐标 (需要 source=/coords= 字段)")

    # (family, chrom, strand) -> [(s, e, evalue)]
    cand = {}
    n_lines = n_missing = n_evalue = 0
    with open(args.mmseqs) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            n_lines += 1
            orf_id, target = cols[0], cols[1]
            ev = None
            if len(cols) >= 3:
                try:
                    ev = float(cols[2])
                except ValueError:
                    ev = None
            if args.max_evalue is not None and ev is not None and ev > args.max_evalue:
                n_evalue += 1
                continue
            coord = orfs.get(orf_id)
            if coord is None:
                n_missing += 1
                continue
            fam = target[:-len(args.strip_suffix)] if args.strip_suffix and target.endswith(args.strip_suffix) else target
            chrom, s, e, strand = coord
            cand.setdefault((fam, chrom, strand), []).append((s, e, ev))

    n_out = n_in = 0
    fams = set()
    with open(args.out, "w") as out:
        out.write("#query_name\tchrom\tstart0\tend0\tstrand\tevalue\n")
        for (fam, chrom, strand) in sorted(cand):
            ivs = sorted(cand[(fam, chrom, strand)], key=lambda x: (x[0], x[1]))
            n_in += len(ivs)
            merged = []
            for s, e, ev in ivs:
                if merged and s <= merged[-1][1] + args.merge:
                    if e > merged[-1][1]:
                        merged[-1][1] = e
                    if ev is not None and (merged[-1][2] is None or ev < merged[-1][2]):
                        merged[-1][2] = ev
                else:
                    merged.append([s, e, ev])
            for s, e, ev in merged:
                out.write(f"{fam}\t{chrom}\t{s}\t{e}\t{strand}\t{'.' if ev is None else f'{ev:.3g}'}\n")
                n_out += 1
            fams.add(fam)

    print(f"[mmseqs2cand] mmseqs 行数 {n_lines}, ORF 坐标缺失 {n_missing}, evalue 过滤 {n_evalue}", file=sys.stderr)
    print(f"[mmseqs2cand] 家族 {len(fams)}, 区间 {n_in} -> 合并后 {n_out}, 写入 {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
