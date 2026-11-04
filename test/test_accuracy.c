#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "mppriv.h"

static void init_complete(mp_reg1_t *reg, mp_extra_t *extra, mp_feat_t *feat, int32_t dp_score)
{
	memset(reg, 0, sizeof(*reg));
	memset(extra, 0, sizeof(*extra));
	memset(feat, 0, 2 * sizeof(*feat));
	extra->dp_max = dp_score;
	extra->dist_start = 0;
	extra->dist_stop = 0;
	reg->qs = 0;
	reg->qe = 100;
	reg->p = extra;
	reg->feat = feat;
	reg->n_feat = reg->n_exon = 2;
	feat[0].type = feat[1].type = MP_FEAT_CDS;
}

static void init_owned(mp_reg1_t *reg, int32_t rank_score, int32_t dp_score)
{
	memset(reg, 0, sizeof(*reg));
	reg->p = (mp_extra_t*)calloc(1, sizeof(*reg->p));
	reg->feat = (mp_feat_t*)calloc(1, sizeof(*reg->feat));
	assert(reg->p != 0 && reg->feat != 0);
	reg->rank_sc = rank_score;
	reg->p->dp_max = dp_score;
}

static void init_ranked(mp_reg1_t *reg, mp_extra_t *extra, int32_t dp_max, int32_t dp_score, uint32_t hash)
{
	memset(reg, 0, sizeof(*reg));
	memset(extra, 0, sizeof(*extra));
	reg->cnt = 1;
	reg->rank_sc = 100;
	reg->hash = hash;
	reg->p = extra;
	extra->dp_max = dp_max;
	extra->dp_score = dp_score;
}

int main(void)
{
	mp_mapopt_t opt;
	mp_reg1_t clean, disrupted, insertion, incumbent, trial;
	mp_extra_t clean_extra, disrupted_extra;
	mp_extra_t ranked_extra[3];
	mp_extra_t *insertion_extra, *accepted_extra;
	mp_feat_t *accepted_feat;
	mp_feat_t clean_feat[2], disrupted_feat[2];
	mp_reg1_t ranked[3];
	int32_t n_ranked;

	mp_mapopt_init(&opt);
	assert(opt.acc_end_retry == 89);
	assert(!(opt.acc_flags & MP_ACC_END_EXT_RETRY)); // rescued candidates skip the io_end 2nd-try by default
	opt.flag |= MP_F_ACCURACY;
	init_complete(&clean, &clean_extra, clean_feat, 100);
	clean_feat[0].donor[0] = 'G', clean_feat[0].donor[1] = 'T';
	clean_feat[1].acceptor[0] = 'A', clean_feat[1].acceptor[1] = 'G';
	mp_set_rank_score(&opt, 100, &clean);
	assert(!mp_reg_needs_refine(&opt, &clean));

	init_complete(&disrupted, &disrupted_extra, disrupted_feat, 110);
	disrupted_extra.n_fs = 1;
	disrupted_extra.n_stop = 1;
	disrupted_feat[0].donor[0] = 'C', disrupted_feat[0].donor[1] = 'C';
	disrupted_feat[1].acceptor[0] = 'C', disrupted_feat[1].acceptor[1] = 'C';
	mp_set_rank_score(&opt, 100, &disrupted);
	assert(mp_reg_needs_refine(&opt, &disrupted));
	assert(mp_reg_score(&clean) > mp_reg_score(&disrupted));

	memset(&insertion, 0, sizeof(insertion));
	insertion_extra = (mp_extra_t*)calloc(1, sizeof(*insertion_extra) + sizeof(uint32_t));
	assert(insertion_extra != 0);
	insertion_extra->n_cigar = 1;
	insertion_extra->cigar[0] = 12U << 4 | NS_CIGAR_I;
	insertion.p = insertion_extra;
	assert(mp_reg_needs_refine(&opt, &insertion));
	insertion_extra->cigar[0] = 31U << 4 | NS_CIGAR_I;
	assert(!mp_reg_needs_refine(&opt, &insertion));
	free(insertion_extra);

	init_owned(&incumbent, 100, 100);
	init_owned(&trial, 99, 200);
	assert(!mp_reg_try_replace(&incumbent, &trial));
	assert(incumbent.rank_sc == 100 && incumbent.p->dp_max == 100);
	assert(trial.p == 0 && trial.feat == 0);

	init_owned(&trial, 100, 101);
	accepted_extra = trial.p, accepted_feat = trial.feat;
	assert(mp_reg_try_replace(&incumbent, &trial));
	assert(incumbent.p == accepted_extra && incumbent.feat == accepted_feat);
	assert(trial.p == 0 && trial.feat == 0);

	init_owned(&trial, 100, 101);
	trial.p->dp_score = 1;
	incumbent.p->dp_score = 0;
	trial.hash = 0;
	incumbent.hash = UINT32_MAX;
	accepted_extra = trial.p, accepted_feat = trial.feat;
	assert(mp_reg_try_replace(&incumbent, &trial));
	assert(incumbent.p == accepted_extra && incumbent.feat == accepted_feat);
	assert(trial.p == 0 && trial.feat == 0);
	free(incumbent.p); free(incumbent.feat);

	init_ranked(&ranked[0], &ranked_extra[0], 90, 100, UINT32_MAX);
	init_ranked(&ranked[1], &ranked_extra[1], 100, 10, UINT32_MAX);
	init_ranked(&ranked[2], &ranked_extra[2], 100, 20, 0);
	n_ranked = 3;
	mp_sort_reg(0, &n_ranked, ranked, 1);
	assert(ranked[0].p == &ranked_extra[2]);
	assert(ranked[1].p == &ranked_extra[1]);
	assert(ranked[2].p == &ranked_extra[0]);
	init_ranked(&ranked[0], &ranked_extra[0], 90, 100, UINT32_MAX);
	init_ranked(&ranked[1], &ranked_extra[1], 100, 10, 0);
	init_ranked(&ranked[2], &ranked_extra[2], 100, 20, 1);
	n_ranked = 3;
	mp_sort_reg(0, &n_ranked, ranked, 0);
	assert(ranked[0].p == &ranked_extra[0]);
	assert(ranked[1].p == &ranked_extra[2]);
	assert(ranked[2].p == &ranked_extra[1]);
	return 0;
}