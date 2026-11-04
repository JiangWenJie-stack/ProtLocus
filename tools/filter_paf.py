#!/usr/bin/env python3
"""Post-filter a ProtLocus PAF file by coverage / completeness / ra:i / fs+st.

Removes low-quality fragmentary hits (typical of --accuracy mode on distant
genome/query pairs) without re-running ProtLocus. Operates line-by-line, so a
query with several PAF records may keep some and drop others.

Examples:
  # drop anything under 50% query coverage
  python3 filter_paf.py --min-cov 0.5 in.paf > out.paf

  # also require a full-length model (both ends complete) and non-negative ra:i
  python3 filter_paf.py --min-cov 0.5 --require-complete --min-ra 0 in.paf > out.paf

  # cap frameshifts/stops per hit
  python3 filter_paf.py --max-fs 2 --max-st 0 in.paf > out.paf
"""
import sys
import re
import argparse


def parse_tags(fields):
    tags = {}
    for f in fields:
        m = re.match(r'^([A-Za-z]{2}):[Aifz]:(.*)$', f)
        if m:
            tags[m.group(1)] = m.group(2)
    return tags


def passes(fields, args):
    qlen, qs, qe, tname = int(fields[1]), int(fields[2]), int(fields[3]), fields[5]
    if tname == '*':
        return True  # pass through unmapped placeholder lines (from -u) untouched
    tags = parse_tags(fields[12:])
    cov = (qe - qs) / qlen if qlen else 0.0
    if cov < args.min_cov:
        return False
    if args.require_complete:
        da, do = int(tags.get('da', -1)), int(tags.get('do', -1))
        if not (da == 0 and do == 0):
            return False
    if args.min_ra is not None and 'ra' in tags and int(tags['ra']) < args.min_ra:
        return False
    fs, st = int(tags.get('fs', 0)), int(tags.get('st', 0))
    if args.max_fs is not None and fs > args.max_fs:
        return False
    if args.max_st is not None and st > args.max_st:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('-o', '--output', default='-', help='output PAF path, "-" for stdout [-]')
    ap.add_argument('--min-cov', type=float, default=0.0, help='min aligned_query/query_length [0.0]')
    ap.add_argument('--require-complete', action='store_true', help='require da==0 and do==0 (both ends complete)')
    ap.add_argument('--min-ra', type=int, default=None, help='min ra:i composite rank score, if present')
    ap.add_argument('--max-fs', type=int, default=None, help='max allowed fs:i frameshift count')
    ap.add_argument('--max-st', type=int, default=None, help='max allowed st:i internal-stop count')
    args = ap.parse_args()

    out = sys.stdout if args.output == '-' else open(args.output, 'w')
    n_in = n_out = 0
    q_in, q_out = set(), set()
    with open(args.input) as fh:
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) < 12:
                continue
            n_in += 1
            q_in.add(f[0])
            if passes(f, args):
                n_out += 1
                q_out.add(f[0])
                out.write(line)
    if args.output != '-':
        out.close()
    print(f"[filter_paf] lines {n_in} -> {n_out} | queries_with_>=1_hit {len(q_in)} -> {len(q_out)}",
          file=sys.stderr)


if __name__ == '__main__':
    main()
