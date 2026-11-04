#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "protlocus.h"
#include "nasw.h"

#define MP_MAX_INTRON_COEF 3.6 // max_intron = sqrt(genome_size) * coef

void mp_idxopt_init(mp_idxopt_t *io)
{
	memset(io, 0, sizeof(*io));
	io->trans_code = MP_CODON_STD;
	io->bbit = 8;
	io->min_aa_len = 30;
	#if MP_BITS_PER_AA == 4
	io->kmer = 6;
	#else
	io->kmer = 8;
	#endif
	io->mod_bit = 1;
}

void mp_mapopt_set_fs(mp_mapopt_t *mo, int32_t fs)
{
	assert(fs >= INT8_MIN && fs <= INT8_MAX);
	mo->fs = fs;
	ns_set_stop_sc(mo->asize, mo->mat, mo->fs);
}

void mp_mapopt_set_max_intron(mp_mapopt_t *mo, int64_t gsize)
{
	int64_t x;
	x = (int64_t)(sqrt((double)gsize) * MP_MAX_INTRON_COEF + 1.);
	if (x < mo->min_max_intron) x = mo->min_max_intron;
	if (x > mo->max_max_intron) x = mo->max_max_intron;
	mo->bw = mo->max_intron = x;
	if (mp_verbose >= 3)
		fprintf(stderr, "[M::%s] set max intron size to %d\n", __func__, mo->max_intron);
}

void mp_mapopt_init(mp_mapopt_t *mo)
{
	memset(mo, 0, sizeof(*mo));
	mo->flag = 0;
	mo->mini_batch_size = 2000000;
	mo->max_occ = 20000;
	mo->max_gap = 1000;
	mo->max_intron = 200000;
	mo->min_max_intron = 10000, mo->max_max_intron = 300000;
	mo->bw = mo->max_intron;
	mo->min_chn_cnt = 3;
	mo->max_chn_max_skip = 25;
	mo->max_chn_iter = 1000000;
	mo->min_chn_sc = 0;
	mo->chn_coef_log = 0.75f;
	mo->max_ext = 10000;
	mo->max_ava = 1000;
	mo->mask_level = 0.5f;
	mo->mask_len = INT32_MAX;
	mo->pri_ratio = 0.7f;
	mo->best_n = 30;
	mo->out_n = 1000;
	mo->out_sim = 0.99f;
	mo->out_cov = 0.1f;
	#if MP_BITS_PER_AA == 4
	mo->kmer2 = 5;
	#else
	mo->kmer2 = 7;
	#endif
	mo->acc_flags = MP_ACC_ALL;
	mo->acc_repeat_cap = 64;
	mo->acc_anchor_budget = 250000;
	mo->acc_rescue_cells = 50000000;
	mo->acc_pri_ratio = 0.35f;
	mo->acc_best_n = 100;
	mo->acc_coarse_n = 0;
	mo->acc_microexon_max = 30;
	mo->acc_cov_bonus = 20;
	mo->acc_splice_bonus = 2;
	mo->acc_noncan_penalty = 6;
	mo->acc_fs_penalty = 8;
	mo->acc_stop_penalty = 12;
	mo->acc_end_bonus = 3;
	mo->acc_rerank_margin = 10;
	mo->acc_end_ext = mo->max_ext;
	mo->acc_end_retry = 89;
	mo->acc_end_retry_win = mo->max_ext;
	mo->acc_end_aa_ext = 100000; // effectively unbounded until tuned; only takes effect with MP_ACC_END_AA_CAP set
	mo->acc_end_ext_min = 300; // nt floor for the dynamic nt-side cap; only takes effect with MP_ACC_END_DYN_WIN set
	mo->acc_end_nt_per_aa = 15.0f;
	mo->acc_end_aa_min = 50; // aa floor for the dynamic aa-side cap; only takes effect with MP_ACC_END_DYN_WIN set
	mo->acc_end_aa_ratio = 0.5f;
	mo->acc_intron_evidence_min = 1000; // nt; only takes effect with MP_ACC_STRICT_MULTI_EXON set
	mo->acc_intron_evidence_ratio = 1.0f; // only takes effect with MP_ACC_STRICT_MULTI_EXON set
	mo->acc_intron_abs_max = 1500; // nt; derived from yeast.strict.paf intron-length distribution (P99=1005, max=2901); only takes effect with MP_ACC_STRICT_MULTI_EXON set
	mo->acc_span_ratio = 0.0f; // off unless set via --acc-strict-span-ratio or the --compact preset
	mo->acc_min_out_score = 0; // off unless set via --acc-min-score or the --compact preset
	mo->acc_min_pos_ratio = 0.0f; // off unless set via --acc-min-pos-ratio or the --compact preset
	mo->acc_rescue_np = 0; // off unless set via --acc-rescue-np or the --compact preset
	mo->acc_rescue_score = 0; // off unless set via --acc-rescue-score
	mo->thr_tab = 0; // off unless --thr-table is given
	mo->cand_tab = 0; // off unless --cand-table is given
	mo->thr_fallback = 24; // 12 bits: yeast decoy sweep showed the fallback is not the binding constraint
	mo->thr_use_ps = 0; // off unless --thr-use-ps is given
	mo->acc_min_ps = 60; // yeast Pfam-A cand-table sweep: AS>=15 (--compact) + ps>=60 dominates every decoy-table operating point; only affects HMM queries (ps); --acc-min-ps=0 disables
	mo->acc_ps_rescue_score = 60, mo->acc_ps_rescue_posr = 0.6f; // yeast Pfam-A B8d sweep: exempting AS>=60 & posr>=0.6 from ps>=60 recovers short-model hits (+22 family-correct genes) for +6% intergenic FP
	mo->cand_max_evalue = 0.0; // no E-value filter on --cand-table unless --tier/--cand-max-evalue is given
	mo->prescreen = 0; // off unless --prescreen is given
	mp_psopt_init(&mo->ps);
	mo->acc_min_np = 0; // off unless set via --acc-min-np or the --compact preset
	mo->hmm_seed_pos_loss = 6; // e^-3: same runner-up tolerance the old single-substitution variants used
	mo->hmm_seed_win_loss = 8; // e^-4 relative to the consensus k-mer's emission probability
	mo->hmm_seed_max = 16;
	mo->hmm_seed_max2 = 16;
	mo->acc_refine_min_seg = 40; // 20 bits: random 5-mer diagonals rarely reach it, true homologous diagonals usually do
	mo->acc_refine_xdrop = 30;
	mo->acc_trim_side_ratio = 5.0f; // a 39-aa exon (117 nt) reached across a 1295-nt intron is cut; two ~100-aa exons across 500 nt are kept
	mo->gpu_device = 0;
	mo->gpu_min_cells = 1000000;
	mo->gpu_prefilter_top_k = 32;
	mo->gpu_prefilter_min_per_group = 1;
	mo->gpu_prefilter_xdrop = 40;

	mo->go = 11, mo->ge = 1;
	mo->io = 29;
	mo->fs = 23;
	mo->io_end = 19;
	mo->ie_coef = .5f;
	mo->sp_model = NS_S_GENERIC;
	mo->sp_null_bonus = -7;
	mo->sp_max_bonus = 14;
	mo->sp_scale = 1.0f;
	mo->end_bonus = 5;
	mo->xdrop = 100;
	mo->asize = 22;
	memcpy(mo->mat, ns_mat_blosum62, 484);
	ns_set_stop_sc(mo->asize, mo->mat, mo->fs);

	mo->gff_delim = -1;
	mo->gff_prefix = "PL";
	mo->hmm = 0;
	mo->max_intron_flank = 200;
}

int32_t mp_mapopt_check(const mp_mapopt_t *mo)
{
	if (mo->sp_model < 0 || mo->sp_model > 2) {
		fprintf(stderr, "[ERROR]\033[1;31m option -j should be between 0 and 2\033[0m\n");
		return -1;
	}
	if (mo->acc_repeat_cap < 0 || mo->acc_anchor_budget < 0 || mo->acc_rescue_cells < 0 ||
		mo->acc_best_n < 0 || mo->acc_microexon_max < 0 || mo->acc_end_ext < 0 || mo->acc_end_retry < 0 ||
		mo->acc_end_aa_ext < 0 || mo->acc_end_ext_min < 0 || mo->acc_end_aa_min < 0 ||
		mo->acc_end_nt_per_aa < 0.0f || mo->acc_end_aa_ratio < 0.0f ||
		mo->acc_intron_evidence_min < 0 || mo->acc_intron_evidence_ratio < 0.0f) {
		fprintf(stderr, "[ERROR]\033[1;31m accuracy budgets must be non-negative\033[0m\n");
		return -1;
	}
	if (mo->acc_pri_ratio < 0.0f || mo->acc_pri_ratio > 1.0f) {
		fprintf(stderr, "[ERROR]\033[1;31m --acc-pri-ratio must be between 0 and 1\033[0m\n");
		return -1;
	}
	if (mo->acc_cov_bonus < 0 || mo->acc_splice_bonus < 0 || mo->acc_noncan_penalty < 0 ||
		mo->acc_fs_penalty < 0 || mo->acc_stop_penalty < 0 || mo->acc_end_bonus < 0 ||
		mo->acc_rerank_margin < 0) {
		fprintf(stderr, "[ERROR]\033[1;31m accuracy ranking weights must be non-negative\033[0m\n");
		return -1;
	}
	if (mo->gpu_device < 0 || mo->gpu_min_cells < 0) {
		fprintf(stderr, "[ERROR]\033[1;31m GPU device and cell threshold must be non-negative\033[0m\n");
		return -1;
	}
	if (mo->gpu_prefilter_top_k < 1 || mo->gpu_prefilter_min_per_group < 0 || mo->gpu_prefilter_xdrop < 0) {
		fprintf(stderr, "[ERROR]\033[1;31m --gpu-prefilter-topk must be >=1 and --gpu-prefilter-diversity/--gpu-prefilter-xdrop must be non-negative\033[0m\n");
		return -1;
	}
	if (mo->gpu_prefilter_min_per_group > mo->gpu_prefilter_top_k) {
		fprintf(stderr, "[ERROR]\033[1;31m --gpu-prefilter-diversity must not exceed --gpu-prefilter-topk\033[0m\n");
		return -1;
	}
	return 0;
}
