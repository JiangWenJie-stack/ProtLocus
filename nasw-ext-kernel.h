/* OpenCL source of the batched end-extension DP; included by nasw-ext-gpu.c.
 *
 * Register-blocked wavefront, resumable in row chunks, several requests per
 * work-group. A work-group has NS_EXT_WG lanes split into 32-lane slots; a request
 * with npad columns takes nb = npad/C blocks and a power-of-two sub-group of
 * >= nb lanes (>= 32), placed at a lane base that is a multiple of its size. Lane
 * Ls of a sub-group owns columns [Ls*C, Ls*C+C) and keeps the three previous rows
 * of H and D plus the A/B/C intron states of those columns in registers. Step t
 * processes the RW rows i = i_beg + RW*(t - Ls) .. i+RW-1 (an anti-diagonal over
 * (row group, block)); the only inter-lane traffic is the last column of the left
 * neighbour (H of rows i-3..i+RW-1, I of rows i..i+RW-1). RW rows per step amortise
 * the per-step fixed cost (exchange, sync, row bookkeeping): a single warp is
 * latency-bound on its own instruction stream, so this is what sets the row rate.
 *
 * Synchronisation (NS_EXT_WARP, inline PTX, NVIDIA only): every sub-group syncs on
 * its own -- bar.warp.sync and shuffles for a 32-lane sub-group, a named barrier and
 * parity-buffered local rows for larger ones -- so each request runs its own step
 * count and leaves the loop when x-drop stops it. Without NS_EXT_WARP the whole
 * work-group barriers once per step and runs the maximum step count.
 *
 * Row group i completes at step (i - i_beg)/RW + nb - 1; every lane of the
 * sub-group then evaluates, row by row, the row maximum, the ie_coef*log2 length
 * penalty, x-drop and the first column reaching the maximum redundantly from
 * local slots, exactly like ns_global_gs16b(). Saturating 16-bit arithmetic is
 * applied in the same order as the CPU kernel; padded columns (al rounded up to
 * the CPU stripe width) are kept because they feed the row maximum on the CPU too.
 *
 * A launch advances every listed request by at most `chunk` rows (a multiple of
 * RW); the DP state (lane registers, request scalars) is saved to / restored from
 * `state`.
 *
 * list: per work-group NS_EXT_WG/32 slots of {request index or -1, sub-group base lane}.
 * Request record (NS_EXT_REQ_INTS ints): nas_off, nl, aas_off, al, npad, io,
 * sp_off, state_off (shorts), row (next row), status (0 running, 1 done),
 * result nt_len, aa_len, score, ok. Per-request state: 9*npad shorts of lane
 * registers, then NS_EXT_STATE_INTS ints of scalars.
 * Local memory: shorts = 2*(2*RW+3)*WG (parity-buffered neighbour rows H and I);
 * ints = 2*RW*(WG + WG/32) (row max / argmax slots, one region per sub-group)
 * + WG/32 (step counts, fallback mode); chars = asize*asize (scoring matrix). */
#ifndef NS_EXT_C
#define NS_EXT_C 4
#endif
#ifndef NS_EXT_RW
#define NS_EXT_RW 4 // rows per step; NS_EXT_CHUNK_ROWS must be a multiple
#endif
#define NS_EXT_STR_(x) #x
#define NS_EXT_STR(x) NS_EXT_STR_(x)
static const char ns_ext_gpu_source[] =
"#define C " NS_EXT_STR(NS_EXT_C) "\n"
"#define RW " NS_EXT_STR(NS_EXT_RW) "\n"
"#define RQ " NS_EXT_STR(NS_EXT_REQ_INTS) "\n"
"#define WG " NS_EXT_STR(NS_EXT_WG) "\n"
"#define NSLOT (WG / 32)\n"
"#define NEG (-32768)\n"
"#define NINF (-0x40000000)\n"
"#ifdef NS_EXT_WARP\n"
"inline int shfl_up(int v) { int r; asm volatile(\"shfl.sync.up.b32 %0, %1, 1, 0, 0xffffffff;\" : \"=r\"(r) : \"r\"(v)); return r; }\n"
"#define SHFL_UP(v) shfl_up(v)\n"
"#define SG_SYNC() do { if (warp) asm volatile(\"bar.warp.sync 0xffffffff;\" ::: \"memory\"); else asm volatile(\"bar.sync %0, %1;\" :: \"r\"(bid), \"r\"(lanes) : \"memory\"); } while (0)\n"
"#define SG_OWN_STEPS 1\n"
"#else\n"
"#define SHFL_UP(v) (v)\n"
"#define SG_SYNC() barrier(CLK_LOCAL_MEM_FENCE)\n"
"#define SG_OWN_STEPS 0\n"
"#endif\n"
"inline int adds(int a, int b) { return clamp(a + b, -32768, 32767); }\n"
"inline int subs(int a, int b) { return clamp(a - b, -32768, 32767); }\n"
"inline char wrap8(int v) { return (char)(((v + 128) & 255) - 128); }\n" // int8_t assignment/compound-assignment wrap-around of the CPU code
/* Device-side ns_ext_prep(): one work-group per newly admitted request turns the raw nt string,
 * optional splice-score track and protein into nas[] (codon -> aa index), aas[], donor[] and
 * acceptor[] exactly as ns_prep_seq()/ns_prep_seq_left() do (left extension reversed + shifted).
 * Every output element is computed from raw inputs only, so there is no phase ordering. tab holds
 * nt4[256], aa20[256], codon[64]; rq[14] bit 0: left extension, bit 1: ss present. */
"__kernel void ns_ext_prep(__global const uchar *rns_all, __global const uchar *rss_all, __global const uchar *ras_all,\n"
"    __global uchar *nas_all, __global uchar *aas_all, __global char *donor_all, __global char *acc_all,\n"
"    __global const int *req, __global const int *plist, __global const uchar *tab,\n"
"    int sp0, int sp1, int sp2, int sp3, int sp4, int sp5, int sp_null) {\n"
"  const int g = plist[get_group_id(0)], L = (int)get_local_id(0), W = (int)get_local_size(0);\n"
"  __global const int *rq = req + g * RQ;\n"
"  const int nl = rq[1], al = rq[3], io = rq[5], left = rq[14] & 1, has_ss = (rq[14] >> 1) & 1;\n"
"  const int max_spsc = (io + 1) / 2 - 1;\n"
"  __global const uchar *ns = rns_all + rq[0], *ss = rss_all + rq[0], *as = ras_all + rq[2];\n"
"  __global uchar *nas = nas_all + rq[0], *aas = aas_all + rq[2];\n"
"  __global char *donor = donor_all + rq[6], *acc = acc_all + rq[6];\n"
"  __global const uchar *nt4 = tab, *aa20 = tab + 256, *codon = tab + 512;\n"
"  const int X = aa20['X'];\n"
"  int sp[6]; sp[0] = sp0; sp[1] = sp1; sp[2] = sp2; sp[3] = sp3; sp[4] = sp4; sp[5] = sp5;\n"
"  for (int j = L; j < al; j += W) aas[j] = aa20[as[left ? al - 1 - j : j]];\n"
"#define NT(p) ((int)nt4[ns[left ? nl - 1 - (p) : (p)]])\n" // the nt4-encoded string as the CPU kernel sees it (reversed for left)
"  for (int i = L; i < nl; i += W) {\n" // codon translation runs over the raw (forward) string; left: reversed then shifted by 2
"    const int k = left ? nl + 1 - i : i;\n"
"    int v = X;\n"
"    if ((!left || i >= 2) && k >= 2) {\n"
"      const int c0 = nt4[ns[k - 2]], c1 = nt4[ns[k - 1]], c2 = nt4[ns[k]];\n"
"      if (c0 < 4 && c1 < 4 && c2 < 4) v = codon[c0 << 4 | c1 << 2 | c2];\n"
"    }\n"
"    nas[i] = (uchar)v;\n"
"  }\n"
"  for (int i = L; i <= nl; i += W) {\n"
"    int d = sp3, a = sp3, s = -1;\n"
"    if (!left) {\n"
"      if (i < nl - 3) {\n"
"        const int n0 = NT(i), n1 = NT(i + 1), n2 = NT(i + 2);\n"
"        int t = 3;\n"
"        if (n1 == 2 && n2 == 3) t = (i + 3 < nl && (NT(i + 3) == 0 || NT(i + 3) == 2)) ? (n0 == 2 ? -1 : 4) : 0;\n" // GT.
"        else if (n1 == 2 && n2 == 1 && n0 == 2) t = 1;\n" // GC.
"        else if (n1 == 0 && n2 == 3) t = 2;\n" // AT.
"        d = t < 0 ? 0 : sp[t];\n"
"      }\n"
"      if (i >= 1 && i < nl) {\n"
"        const int nm1 = NT(i - 1), n0 = NT(i);\n"
"        int t = 3, penY = 0;\n"
"        if (nm1 == 0 && n0 == 2) {\n" // .AG
"          t = (i >= 2 && (NT(i - 2) == 1 || NT(i - 2) == 3)) ? -1 : 0;\n"
"          for (int j = i - 4; j >= 0 && j > i - 7; --j) if (NT(j) != 1 && NT(j) != 3) penY += sp5;\n"
"        } else if (nm1 == 0 && n0 == 1) t = 2;\n" // .AC
"        a = t < 0 ? 0 : sp[t];\n"
"        if (t == -1 || t == 0) a += penY;\n"
"      }\n"
"      if (has_ss && i + 1 < nl) s = ss[i + 1];\n" // ss[p] adjusts position p-1
"      if (s >= 0) {\n"
"        int spsc = min((s >> 1) - 64, max_spsc);\n"
"        if (s == 0xff) { d -= sp_null; a -= sp_null; } else if (s & 1) a -= spsc; else d -= spsc;\n"
"      }\n"
"    } else {\n"
"      if (i < nl - 3) {\n"
"        const int n1 = NT(i + 1), n2 = NT(i + 2);\n"
"        int t = 3, penY = 0;\n"
"        if (n1 == 2 && n2 == 0) {\n" // GA. (reverse of .AG)
"          t = (i + 3 < nl && (NT(i + 3) == 1 || NT(i + 3) == 3)) ? -1 : 0;\n"
"          for (int j = i + 5; j < nl && j < i + 8; ++j) if (NT(j) != 1 && NT(j) != 3) penY += sp5;\n"
"        } else if (n1 == 1 && n2 == 0) t = 2;\n" // CA.
"        d = t < 0 ? 0 : sp[t];\n"
"        if (t == -1 || t == 0) d += penY;\n"
"      }\n"
"      if (i >= 1 && i < nl) {\n"
"        const int nm1 = NT(i - 1), n0 = NT(i);\n"
"        int t = 3;\n"
"        if (nm1 == 3 && n0 == 2) t = (i >= 2 && (NT(i - 2) == 0 || NT(i - 2) == 2)) ? (i + 1 < nl && NT(i + 1) == 2 ? -1 : 4) : 0;\n" // .TG
"        else if (nm1 == 1 && n0 == 2 && i + 1 < nl && NT(i + 1) == 1) t = 1;\n" // .CG
"        else if (nm1 == 3 && n0 == 0) t = 2;\n" // .TA
"        a = t < 0 ? 0 : sp[t];\n"
"      }\n"
"      if (has_ss && i < nl) s = ss[nl - 1 - i];\n" // ss[p] adjusts position nl-1-p
"      if (s >= 0) {\n"
"        int spsc = min((s >> 1) - 64, max_spsc);\n"
"        if (s == 0xff) { d -= sp_null; a -= sp_null; } else if (s & 1) d -= spsc; else a -= spsc;\n"
"      }\n"
"    }\n"
"    donor[i] = wrap8(d); acc[i] = wrap8(a);\n"
"  }\n"
"#undef NT\n"
"}\n"
"__attribute__((reqd_work_group_size(WG, 1, 1)))\n"
"__kernel void ns_ext_dp(__global const uchar *nas_all, __global const uchar *aas_all,\n"
"    __global const char *donor_all, __global const char *acc_all,\n"
"    __global int *req, __global const int *list, __global short *state,\n"
"    __global const char *mat_g, __global const int *pen,\n"
"    int go, int ge, int fs, int xdrop, int end_bonus, int asize, int chunk,\n"
"    __local short *sm, __local int *im, __local char *mat) {\n"
"  const int L = (int)get_local_id(0), wg = (int)get_group_id(0);\n"
"  const int g = list[(wg * NSLOT + L / 32) * 2], base = list[(wg * NSLOT + L / 32) * 2 + 1];\n"
"  const int Ls = L - base, Lm = max(L - 1, 0);\n"
"  __local short *bH = sm, *bI = sm + 2 * (RW + 3) * WG;\n" // [parity][row][lane]
"  __local int *steps_l = im + 2 * RW * (WG + NSLOT);\n"
"  __local int *rowmax = im + RW * (base + base / 32), *rowkey = rowmax + RW * (WG + NSLOT);\n" // sub-group region, RW slots per group
"  for (int k = L; k < asize * asize; k += WG) mat[k] = mat_g[k];\n"
"  for (int k = L; k < 2 * RW * (WG + NSLOT); k += WG) im[k] = k < RW * (WG + NSLOT) ? NINF : -1;\n"
"  if (L < NSLOT) steps_l[L] = 0;\n"
"  __global int *rq = req + (g < 0 ? 0 : g) * RQ;\n"
"  const int nl = rq[1], al = rq[3], npad = rq[4], io = rq[5], i_beg = rq[8];\n"
"  __global const uchar *nas = nas_all + rq[0], *aas = aas_all + rq[2];\n"
"  __global const char *donor = donor_all + rq[6], *acceptor = acc_all + rq[6];\n"
"  const int nb = npad / C, R = nb + 1, j0 = Ls * C;\n"
"  const int active = g >= 0 && Ls < nb;\n"
"  const int i_end = i_beg + chunk < nl ? i_beg + chunk : nl;\n"
"  const int own_steps = g >= 0 ? (i_end - i_beg + RW - 1) / RW + nb - 1 : 0;\n"
"  int lanes = 32;\n" // sub-group size, as ns_ext_lanes_for() computes it
"  while (lanes < nb) lanes <<= 1;\n"
"  const int warp = SG_OWN_STEPS && lanes == 32, bid = base / 32 + 1;\n" // named barrier ids 1..NSLOT; 0 is barrier()
"  __global short *st = state + rq[7] + Ls * 9 * C;\n"
"  __global int *sc = (__global int*)(state + rq[7] + nb * 9 * C);\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  if (g >= 0 && Ls == 0) steps_l[L / 32] = own_steps;\n"
"  int H1[C], H2[C], H3[C], D1[C], D2[C], D3[C], A[C], B[C], Cc[C], aj[C], ajc[C], smask[C], sneg[C];\n"
"  int max_sc, max_sc_log, max_i, first_j;\n"
"  if (g < 0 || i_beg == 2) {\n"
"    for (int k = 0; k < C; ++k) H1[k] = H2[k] = H3[k] = D1[k] = D2[k] = D3[k] = A[k] = B[k] = Cc[k] = NEG;\n"
"    max_sc = max_sc_log = -0x7fffffff - 1; max_i = -1; first_j = -1;\n"
"  } else {\n"
"    if (active) for (int k = 0; k < C; ++k) {\n"
"      H1[k] = st[k]; H2[k] = st[C + k]; H3[k] = st[2 * C + k];\n"
"      D1[k] = st[3 * C + k]; D2[k] = st[4 * C + k]; D3[k] = st[5 * C + k];\n"
"      A[k] = st[6 * C + k]; B[k] = st[7 * C + k]; Cc[k] = st[8 * C + k];\n"
"    }\n"
"    max_sc = sc[0]; max_sc_log = sc[1]; max_i = sc[2]; first_j = sc[3];\n"
"  }\n"
"  for (int k = 0; k < C; ++k) {\n" // padded columns score NEG: (mat & smask) | sneg keeps the load unconditional
"    aj[k] = (active && j0 + k < al) ? (int)aas[j0 + k] : -1;\n"
"    ajc[k] = max(aj[k], 0); smask[k] = aj[k] >= 0 ? -1 : 0; sneg[k] = aj[k] >= 0 ? 0 : NEG;\n"
"  }\n"
"  int hL[RW + 3], iL[RW];\n" // my last column: H of rows i-3..i+RW-1 and I of rows i..i+RW-1 of the group just done
"  for (int q = 0; q < RW + 3; ++q) hL[q] = NEG;\n"
"  for (int q = 0; q < RW; ++q) iL[q] = NEG;\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  int steps = own_steps;\n"
"  if (!SG_OWN_STEPS) { steps = 0; for (int k = 0; k < NSLOT; ++k) steps = max(steps, steps_l[k]); }\n"
"  const int pen_len = al * 3;\n"
"  int done = 0;\n"
/* Ring positions are carried as counters (group index mod R): the step body is kept free of
 * divisions and of data-dependent branches so a lone warp streams through it. */
"  int ir = (-Ls) % R; if (ir < 0) ir += R;\n"
"  int rr = (1 - nb) % R; if (rr < 0) rr += R;\n"
"  for (int t = 0; t < steps; ++t) {\n"
"    const int i = i_beg + RW * (t - Ls);\n" // first row of my group this step
"    int nh[RW + 3], ni[RW];\n"
"    if (warp) {\n"
"      for (int q = 0; q < RW + 3; ++q) nh[q] = SHFL_UP(hL[q]);\n"
"      for (int q = 0; q < RW; ++q) ni[q] = SHFL_UP(iL[q]);\n"
"    }\n"
"    if (active && !done && i >= i_beg && i < i_end) {\n"
"      if (!warp) {\n" // written by the left lane at step t-1 into parity (t-1)&1
"        for (int q = 0; q < RW + 3; ++q) nh[q] = bH[(((t + 1) & 1) * (RW + 3) + q) * WG + Lm];\n"
"        for (int q = 0; q < RW; ++q) ni[q] = bI[(((t + 1) & 1) * RW + q) * WG + Lm];\n"
"      }\n"
"      hL[0] = H3[C - 1]; hL[1] = H2[C - 1]; hL[2] = H1[C - 1];\n"
"      int an[RW], dn[RW + 2], cn[RW + 2];\n" // rows past i_end are computed on clamped inputs and ignored
"      for (int q = 0; q < RW; ++q) an[q] = nas[min(i + q, nl - 1)];\n"
"      for (int q = 0; q < RW + 2; ++q) { dn[q] = donor[min(i - 1 + q, nl)]; cn[q] = acceptor[min(i - 2 + q, nl)]; }\n"
"      for (int q = 0; q < RW; ++q) {\n"
"        const int iq = i + q, a_i = an[q], gei = a_i == 20 ? fs : ge;\n"
"        __local const char *mrow = mat + a_i * asize;\n"
"        const int dm1 = dn[q], d0 = dn[q + 1], dp1 = dn[q + 2], am2 = cn[q], am1 = cn[q + 1], a0 = cn[q + 2];\n"
"        int hl3 = Ls == 0 ? (iq == 2 ? 0 : NEG) : nh[q], hl2 = Ls == 0 ? (iq == 2 ? -fs : NEG) : nh[q + 1];\n"
"        int hl1 = Ls == 0 ? (iq == 2 ? -fs : NEG) : nh[q + 2], hl = Ls == 0 ? NEG : nh[q + 3], il = Ls == 0 ? NEG : ni[q];\n"
"        int m = NINF, key = -1;\n"
"        int Hc[C], Dc[C];\n"
"        for (int k = 0; k < C; ++k) {\n"
"          const int j = j0 + k;\n"
"          int s = ((int)mrow[ajc[k]] & smask[k]) | sneg[k];\n"
"          int h = adds(hl3, s);\n"
"          int x = subs(max(subs(H3[k], go), D3[k]), gei);\n"
"          Dc[k] = x; h = max(h, x);\n"
"          int h1 = H1[k], u = subs(h1, io);\n"
"          x = max(subs(u, dm1), A[k]); A[k] = x;\n"
"          h = max(h, subs(x, a0)); h = max(h, subs(h1, fs));\n"
"          u = subs(hl1, io);\n"
"          x = max(subs(u, d0), B[k]); B[k] = x; h = max(h, subs(x, am2));\n"
"          x = max(subs(u, dp1), Cc[k]); Cc[k] = x; h = max(h, subs(x, am1));\n"
"          h = max(h, subs(hl1, fs));\n"
"          h = max(h, subs(H2[k], fs)); h = max(h, subs(hl2, fs));\n"
"          il = subs(max(subs(hl, go), il), ge); h = max(h, il);\n"
"          Hc[k] = h; hl = h; hl3 = H3[k]; hl2 = H2[k]; hl1 = H1[k];\n"
"          { const int v = h + (j == al - 1 ? end_bonus : 0); m = max(m, max(h, v));\n"
"            key = max(key, j < al ? (((v + 32768 + 128) << 12) | (4095 - j)) : -1); }\n"
"        }\n"
"        hL[3 + q] = hl; iL[q] = il;\n"
"        atomic_max(&rowmax[ir * RW + q], m); atomic_max(&rowkey[ir * RW + q], key);\n"
"        for (int k = 0; k < C; ++k) { H3[k] = H2[k]; H2[k] = H1[k]; H1[k] = Hc[k]; D3[k] = D2[k]; D2[k] = D1[k]; D1[k] = Dc[k]; }\n"
"      }\n"
"      if (!warp) {\n"
"        for (int q = 0; q < RW + 3; ++q) bH[((t & 1) * (RW + 3) + q) * WG + L] = (short)hL[q];\n"
"        for (int q = 0; q < RW; ++q) bI[((t & 1) * RW + q) * WG + L] = (short)iL[q];\n"
"      }\n"
"    }\n"
"    ir = ir + 1 == R ? 0 : ir + 1;\n"
"    SG_SYNC();\n"
"    if (g >= 0 && !done) {\n" // sub-group uniform
"      const int rg = i_beg + RW * (t - nb + 1);\n" // first row of the group that has just completed
"      for (int q = 0; q < RW; ++q) {\n"
"        const int r = rg + q, act = !done && r >= i_beg && r < i_end;\n"
"        const int tmp_sc = rowmax[rr * RW + q], k2 = rowkey[rr * RW + q];\n"
"        const int tmp_sc_log = tmp_sc - pen[clamp(r - pen_len, 0, nl - 1)];\n" // pen[0] = pen[1] = 0
"        const int upd = act && tmp_sc_log > max_sc_log;\n"
"        max_sc = upd ? tmp_sc : max_sc; max_i = upd ? r : max_i;\n"
"        first_j = upd ? ((k2 >> 12) - 32768 - 128 == tmp_sc ? 4095 - (k2 & 4095) : -1) : first_j;\n"
"        max_sc_log = upd ? tmp_sc_log : max_sc_log;\n"
"        done = done || (act && max_sc_log - tmp_sc_log > xdrop);\n"
"      }\n"
"      if (Ls == 0 && rg - 1 >= i_beg) { const int rp = rr ? rr - 1 : R - 1; for (int q = 0; q < RW; ++q) rowmax[rp * RW + q] = NINF, rowkey[rp * RW + q] = -1; }\n" // same lane writes that slot next step
"      if (SG_OWN_STEPS && done) break;\n" // uniform within the sub-group, which syncs on its own
"    }\n"
"    rr = rr + 1 == R ? 0 : rr + 1;\n"
"  }\n"
"  if (g < 0) return;\n"
"  if (i_end >= nl) done = 1;\n"
"  if (!done && active) for (int k = 0; k < C; ++k) {\n"
"    st[k] = (short)H1[k]; st[C + k] = (short)H2[k]; st[2 * C + k] = (short)H3[k];\n"
"    st[3 * C + k] = (short)D1[k]; st[4 * C + k] = (short)D2[k]; st[5 * C + k] = (short)D3[k];\n"
"    st[6 * C + k] = (short)A[k]; st[7 * C + k] = (short)B[k]; st[8 * C + k] = (short)Cc[k];\n"
"  }\n"
"  if (Ls == 0) {\n"
"    if (done) {\n"
"      rq[9] = 1; rq[10] = max_i + 1; rq[11] = first_j + 1; rq[12] = max_sc; rq[13] = (max_i >= 0 && first_j >= 0);\n"
"    } else {\n"
"      rq[8] = i_end; sc[0] = max_sc; sc[1] = max_sc_log; sc[2] = max_i; sc[3] = first_j;\n"
"    }\n"
"  }\n"
"}\n";
