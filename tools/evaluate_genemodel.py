#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate_genemodel.py -- 把 protlocus_genemodel.py 的 GFF3 与参考注释 (RefSeq/TAIR GFF3) 比较。

指标 (均按 CDS, 含终止子):
  nt 级敏感性/特异性; 外显子级(CDS 区间完全一致)敏感性/精确度;
  基因级: 预测与参考某转录本 CDS 链完全一致 / 与参考 CDS 同链重叠;
  同时给出 "有证据覆盖的参考基因" 子集上的召回(参考基因与任一命中重叠)。
"""
from __future__ import print_function
import argparse
import collections
import gzip
import re
import sys


def open_text(p):
    return gzip.open(p, 'rt') if p.endswith('.gz') else open(p)


def read_gff(path, cds_types=('CDS',)):
    """返回 {transcript_id: (contig, strand, [(s,e)] 0-based 半开, 排序)}"""
    tx = collections.defaultdict(list)
    meta = {}
    with open_text(path) as fh:
        for line in fh:
            if line[0] == '#':
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 9 or f[2] not in cds_types:
                continue
            m = re.search(r'Parent=([^;]+)', f[8])
            if not m:
                continue
            pid = m.group(1).split(',')[0]
            tx[pid].append((int(f[3]) - 1, int(f[4])))
            meta[pid] = (f[0], f[6])
    out = {}
    for k, v in tx.items():
        out[k] = (meta[k][0], meta[k][1], sorted(v))
    return out


def merge_iv(ivs):
    ivs = sorted(ivs)
    out = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total_len(ivs):
    return sum(e - s for s, e in ivs)


def inter_len(a, b):
    i = j = t = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            t += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--paf', default=None, help='final.paf, 用于统计"有证据覆盖"的参考基因')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    ref = read_gff(a.ref)
    pred = read_gff(a.pred)
    # 参考基因: 按 CDS 并集聚类同一 contig/strand 重叠的转录本
    by = collections.defaultdict(list)
    for tid, (c, s, ex) in ref.items():
        by[(c, s)].append((ex[0][0], ex[-1][1], tid))
    genes = []
    for key, lst in by.items():
        lst.sort()
        cur = None
        for s, e, tid in lst:
            if cur and s < cur['e']:
                cur['e'] = max(cur['e'], e)
                cur['tx'].append(tid)
            else:
                cur = {'key': key, 's': s, 'e': e, 'tx': [tid]}
                genes.append(cur)
    # 预测基因
    pby = collections.defaultdict(list)
    for tid, (c, s, ex) in pred.items():
        pby[(c, s)].append((ex[0][0], ex[-1][1], tid))
    for k in pby:
        pby[k].sort()

    # nt 级
    nt_ref = nt_pred = nt_tp = 0
    ref_nt = collections.defaultdict(list)
    pred_nt = collections.defaultdict(list)
    for tid, (c, s, ex) in ref.items():
        ref_nt[(c, s)].extend(ex)
    for tid, (c, s, ex) in pred.items():
        pred_nt[(c, s)].extend(ex)
    for k in set(ref_nt) | set(pred_nt):
        r, p = merge_iv(ref_nt.get(k, [])), merge_iv(pred_nt.get(k, []))
        nt_ref += total_len(r)
        nt_pred += total_len(p)
        nt_tp += inter_len(r, p)

    # 外显子级 (参考外显子唯一集合 vs 预测外显子)
    rex = set()
    pex = set()
    for tid, (c, s, ex) in ref.items():
        for x in ex:
            rex.add((c, s, x))
    for tid, (c, s, ex) in pred.items():
        for x in ex:
            pex.add((c, s, x))
    ex_tp_r = len(rex & pex)

    # 链完全一致
    rchains = collections.defaultdict(set)
    for tid, (c, s, ex) in ref.items():
        rchains[(c, s)].add(tuple(ex))
    exact_pred = 0
    for tid, (c, s, ex) in pred.items():
        if tuple(ex) in rchains[(c, s)]:
            exact_pred += 1
    pchains = collections.defaultdict(set)
    for tid, (c, s, ex) in pred.items():
        pchains[(c, s)].add(tuple(ex))
    exact_ref_genes = 0
    detected = 0
    split_ref = 0
    import bisect
    ref_overlap_genes = {}
    for gi, g in enumerate(genes):
        key = g['key']
        plist = pby.get(key, [])
        starts = [p[0] for p in plist]
        hit = []
        i = bisect.bisect_left(starts, g['s'])
        j = max(0, i - 50)
        for p in plist[j:]:
            if p[0] >= g['e']:
                break
            if p[1] > g['s']:
                # CDS 级重叠检查
                pex_ = pred[p[2]][2]
                gex = []
                for t in g['tx']:
                    gex.extend(ref[t][2])
                if inter_len(merge_iv(gex), merge_iv(pex_)) > 0:
                    hit.append(p[2])
        if hit:
            detected += 1
            if len(hit) > 1:
                split_ref += 1
        g['hit'] = hit
        if any(tuple(ref[t][2]) in pchains[key] for t in g['tx']):
            exact_ref_genes += 1

    # 预测基因: 与参考重叠 / 合并多个参考基因
    gene_starts = collections.defaultdict(list)
    for g in genes:
        gene_starts[g['key']].append(g)
    pred_overlap = 0
    merged = 0
    for tid, (c, s, ex) in pred.items():
        n = 0
        for g in gene_starts.get((c, s), []):
            if tid in g['hit']:
                n += 1
        if n:
            pred_overlap += 1
        if n > 1:
            merged += 1

    covered = None
    if a.paf:
        cov = collections.defaultdict(list)
        with open(a.paf) as fh:
            for line in fh:
                if line[0] == '#':
                    continue
                f = line.split('\t', 9)
                if len(f) < 9 or f[4] not in '+-':
                    continue
                cov[(f[5], f[4])].append((int(f[7]), int(f[8])))
        for k in cov:
            cov[k].sort()
        covered = 0
        cov_det = 0
        cov_exact = 0
        for g in genes:
            lst = cov.get(g['key'], [])
            st = [x[0] for x in lst]
            i = bisect.bisect_left(st, g['e'])
            ok = False
            for s, e in lst[max(0, i - 400):i]:
                if e > g['s'] and s < g['e']:
                    ok = True
                    break
            if ok:
                covered += 1
                if g['hit']:
                    cov_det += 1
                if any(tuple(ref[t][2]) in pchains[g['key']] for t in g['tx']):
                    cov_exact += 1

    ng = len(genes)
    print('参考: %d 基因 / %d 转录本 / %d CDS nt; 预测: %d 基因' % (ng, len(ref), nt_ref, len(pred)))
    print('nt 级:     敏感性 %.4f  特异性 %.4f' % (nt_tp / float(nt_ref or 1), nt_tp / float(nt_pred or 1)))
    print('外显子级:  敏感性 %.4f  精确度 %.4f   (参考 %d / 预测 %d 外显子)' % (
        ex_tp_r / float(len(rex) or 1), ex_tp_r / float(len(pex) or 1), len(rex), len(pex)))
    print('基因级:    参考基因被检出(同链 CDS 重叠) %d/%d = %.4f; CDS 链完全一致 %d = %.4f' % (
        detected, ng, detected / float(ng or 1), exact_ref_genes, exact_ref_genes / float(ng or 1)))
    print('           预测基因与参考重叠 %d/%d = %.4f; 预测链与某参考转录本完全一致 %d = %.4f' % (
        pred_overlap, len(pred), pred_overlap / float(len(pred) or 1), exact_pred,
        exact_pred / float(len(pred) or 1)))
    print('           预测合并了多个参考基因: %d; 参考基因被拆成多个预测: %d' % (merged, split_ref))
    if covered is not None:
        print('命中覆盖的参考基因 %d (%.4f of 参考): 其中检出 %.4f, CDS 链完全一致 %.4f' % (
            covered, covered / float(ng or 1), cov_det / float(covered or 1), cov_exact / float(covered or 1)))


if __name__ == '__main__':
    main()
