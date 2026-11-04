#!/usr/bin/env python3
"""Generic PAF quality/recall metrics and baseline-vs-candidate comparison.

Ground-truth-free proxy metrics (coverage, complete-both-ends, fs/st, ra:i),
generalizing the ad hoc benchmark_results/parse_paf.py and compare_paf.py
scripts to arbitrary file paths so the same analysis can be reused for
candidate-retention sweeps (--outc/--acc-best-n/--acc-pri-ratio) and
sub-flag ablations (--acc-no-*).

Examples:
  python3 paf_metrics.py summary mouse_baseline.paf mouse_accuracy.paf

  python3 paf_metrics.py compare --baseline frog_baseline.paf \\
      frog_accuracy.paf frog_accuracy_tight.paf frog_accuracy_repeatonly.paf
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


def load_best(path):
    """Best (highest AS) PAF record per query. Returns (best dict, all query names)."""
    best = {}
    all_q = set()
    with open(path) as fh:
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) < 12:
                continue
            qname, qlen, qs, qe, tname = f[0], int(f[1]), int(f[2]), int(f[3]), f[5]
            all_q.add(qname)
            if tname == '*':
                continue
            tags = parse_tags(f[12:])
            score = int(tags.get('AS', -1))
            cov = (qe - qs) / qlen if qlen else 0.0
            da, do = int(tags.get('da', -1)), int(tags.get('do', -1))
            complete = (da == 0 and do == 0)
            fs, st = int(tags.get('fs', 0)), int(tags.get('st', 0))
            ra = int(tags['ra']) if 'ra' in tags else None
            rec = (score, cov, complete, fs, st, ra)
            if qname not in best or score > best[qname][0]:
                best[qname] = rec
    return best, all_q


def summary(path):
    best, all_q = load_best(path)
    n_total, n_mapped = len(all_q), len(best)
    cov_sum = sum(r[1] for r in best.values())
    complete = sum(1 for r in best.values() if r[2])
    fs_total = sum(r[3] for r in best.values())
    st_total = sum(r[4] for r in best.values())
    ra_vals = [r[5] for r in best.values() if r[5] is not None]
    return {
        'n_total': n_total, 'n_mapped': n_mapped,
        'recall_pct': 100.0 * n_mapped / n_total if n_total else 0.0,
        'avg_cov': cov_sum / n_mapped if n_mapped else 0.0,
        'complete_pct': 100.0 * complete / n_mapped if n_mapped else 0.0,
        'fs_total': fs_total, 'st_total': st_total,
        'avg_ra': (sum(ra_vals) / len(ra_vals)) if ra_vals else None,
    }


def cmd_summary(args):
    for path in args.files:
        r = summary(path)
        ra_str = f"{r['avg_ra']:.1f}" if r['avg_ra'] is not None else 'n/a'
        print(f"{path}: mapped {r['n_mapped']}/{r['n_total']} ({r['recall_pct']:.1f}%) "
              f"avg_cov={r['avg_cov']:.3f} complete={r['complete_pct']:.1f}% "
              f"fs={r['fs_total']} st={r['st_total']} avg_ra={ra_str}")


def cmd_compare(args):
    base, all_q = load_best(args.baseline)
    for path in args.candidates:
        cand, cand_all_q = load_best(path)
        union_q = all_q | cand_all_q
        both = set(base) & set(cand)
        only_base = set(base) - set(cand)
        only_cand = set(cand) - set(base)
        print(f"\n=== {path} vs baseline {args.baseline} (total queries={len(union_q)}) ===")
        print(f"  mapped by both: {len(both)} | only baseline (regressions): {len(only_base)} "
              f"| only candidate (new recall): {len(only_cand)}")
        if both:
            n_better = sum(1 for q in both if cand[q][0] > base[q][0])
            n_worse = sum(1 for q in both if cand[q][0] < base[q][0])
            cov_delta = sum(cand[q][1] - base[q][1] for q in both) / len(both)
            comp_b = sum(1 for q in both if base[q][2])
            comp_c = sum(1 for q in both if cand[q][2])
            fs_b = sum(base[q][3] for q in both); fs_c = sum(cand[q][3] for q in both)
            st_b = sum(base[q][4] for q in both); st_c = sum(cand[q][4] for q in both)
            print(f"  [shared] score better={n_better} worse={n_worse} | avg cov delta={cov_delta:+.4f}")
            print(f"  [shared] complete: base={100*comp_b/len(both):.1f}% cand={100*comp_c/len(both):.1f}% "
                  f"| fs: base={fs_b} cand={fs_c} | st: base={st_b} cand={st_c}")
        if only_cand:
            covs = [cand[q][1] for q in only_cand]
            comps = [cand[q][2] for q in only_cand]
            fs_only = sum(cand[q][3] for q in only_cand)
            st_only = sum(cand[q][4] for q in only_cand)
            print(f"  [candidate-only new hits] n={len(only_cand)} avg_cov={sum(covs)/len(covs):.3f} "
                  f"complete={100*sum(comps)/len(comps):.1f}% fs={fs_only} st={st_only}")


def cmd_regressions(args):
    """List exact queries lost (only-baseline) or scored worse than baseline, to trace which
    --acc-no-* ablation flips them back."""
    base, _ = load_best(args.baseline)
    cand, _ = load_best(args.candidate)
    only_base = sorted(set(base) - set(cand))
    worse = sorted((q for q in (set(base) & set(cand)) if cand[q][0] < base[q][0]),
                   key=lambda q: cand[q][0] - base[q][0])

    tab = '\t'
    print(f"=== lost (only baseline) n={len(only_base)} ===")
    print(tab.join(['query', 'base_AS', 'base_cov', 'base_fs', 'base_st']))
    for q in only_base[:args.limit]:
        s, cov, comp, fs, st, ra = base[q]
        print(tab.join([q, str(s), f"{cov:.3f}", str(fs), str(st)]))

    print(f"\n=== shared but scored worse n={len(worse)} (sorted by largest AS drop first) ===")
    print(tab.join(['query', 'base_AS', 'cand_AS', 'delta', 'base_cov', 'cand_cov']))
    for q in worse[:args.limit]:
        bs, bcov = base[q][0], base[q][1]
        cs, ccov = cand[q][0], cand[q][1]
        print(tab.join([q, str(bs), str(cs), str(cs - bs), f"{bcov:.3f}", f"{ccov:.3f}"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('summary', help='per-file recall/coverage/completeness/fs/st/ra summary')
    sp.add_argument('files', nargs='+')
    sp.set_defaults(func=cmd_summary)

    sp = sub.add_parser('compare', help='baseline vs one or more candidate PAFs (recall gain/regression, shared-hit quality)')
    sp.add_argument('--baseline', required=True)
    sp.add_argument('candidates', nargs='+')
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser('regressions', help='list exact queries lost or scored worse than baseline')
    sp.add_argument('--baseline', required=True)
    sp.add_argument('--limit', type=int, default=50, help='max rows to print per section')
    sp.add_argument('candidate')
    sp.set_defaults(func=cmd_regressions)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
