#!/usr/bin/env bash
## ProtLocus「候选 + HMM 复核」一键流水线
##
##   第 1 段  protlocus --prescreen --tier=... --trans  →  候选 PAF（每行命中后跟
##            ##STA 剪接后翻译肽段）
##   第 2 段  protlocus-recheck（原生 C 复核器，默认）→  每条候选肽段只对它自己
##            家族的 profile 跑完整 Forward 算法，用 profile 自带的
##            STATS LOCAL FORWARD 校准参数换算 E 值并过滤（默认 1e-5）。
##            全程不依赖 HMMER；-R hmmer 可切换到 hmmrecheck.py（调用
##            hmmfetch/hmmsearch，用于交叉验证）。
##
## protlocus 参数默认取基准冻结值：-I -u --outs=0 -N 1000 --outn=1000
## --acc-pri-ratio=0.1；大基因组 --accuracy、小基因组可 -M compact。
##
## 用法:
##   protlocus_hmm_pipeline.sh -g genome.fa[.mpi] -m Pfam-A.hmm -o OUTPREFIX \
##       [-b protlocus] [-r protlocus-recheck] [-R native|hmmer (默认 native)] \
##       [-t 线程=16] [-T fast|balanced|sensitive (默认 balanced)] \
##       [-M accuracy|compact (默认 accuracy)] [-E 1e-5] [-Z 有效库大小] \
##       [-x "额外 protlocus 选项"] \
##       [-G genome.fa [-X "genemodel 额外选项"]]   # 第 3 段, 由命中构建基因模型
##
## 产物:
##   OUTPREFIX.cand.paf     第 1 段候选（PAF + ##STA）
##   OUTPREFIX.final.paf    复核通过的最终 PAF（native: fB:f:/fE:f:；hmmer: hB:f:/hE:f:）
##   OUTPREFIX.recheck.tsv  每条候选的复核明细
##   OUTPREFIX.stage{1,2}.log  各段日志（含 /usr/bin/time -v 资源统计）
##   -G 时另有: OUTPREFIX.genes.gff3 (基因模型 GFF3)、OUTPREFIX.genes.faa (蛋白)、OUTPREFIX.stage3.log
##   第 3 段 tools/protlocus_genemodel.py: 合并同位点命中、ORF 延伸到起止密码子、连接区剪接搜索;
##   -G 给基因组 FASTA (非 .mpi); 原核基因组请用 -X "--no-introns"。
set -eu

BIN=protlocus; THREADS=16; TIER=balanced; MODE=accuracy; ENGINE=native
MAX_IE=1e-5; ZVAL=""; EXTRA=""; GENOME=""; HMM=""; PREFIX=""; RECHECK_BIN=""; GENOME_FA=""; GM_EXTRA=""
while getopts "g:m:o:b:r:R:t:T:M:E:Z:x:G:X:h" opt; do
  case $opt in
    g) GENOME=$OPTARG;; m) HMM=$OPTARG;; o) PREFIX=$OPTARG;;
    b) BIN=$OPTARG;; r) RECHECK_BIN=$OPTARG;; R) ENGINE=$OPTARG;;
    t) THREADS=$OPTARG;; T) TIER=$OPTARG;;
    M) MODE=$OPTARG;; E) MAX_IE=$OPTARG;; Z) ZVAL=$OPTARG;; x) EXTRA=$OPTARG;;
    G) GENOME_FA=$OPTARG;; X) GM_EXTRA=$OPTARG;;
    h|*) grep '^##' "$0" | sed 's/^## \{0,1\}//'; exit 0;;
  esac
done
[ -n "$GENOME" ] && [ -n "$HMM" ] && [ -n "$PREFIX" ] || {
  echo "[FATAL] 必须给 -g 基因组 -m HMM 库 -o 输出前缀（-h 看用法）" >&2; exit 1; }
[ -s "$GENOME" ] || { echo "[FATAL] 基因组不存在: $GENOME" >&2; exit 1; }
[ -s "$HMM" ]    || { echo "[FATAL] HMM 库不存在: $HMM" >&2; exit 1; }
command -v "$BIN" >/dev/null || { echo "[FATAL] 找不到 protlocus: $BIN" >&2; exit 1; }
case $MODE in accuracy|compact) ;; *) echo "[FATAL] -M 只能是 accuracy|compact" >&2; exit 1;; esac
case $ENGINE in native|hmmer) ;; *) echo "[FATAL] -R 只能是 native|hmmer" >&2; exit 1;; esac

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
if [ "$ENGINE" = native ]; then
  if [ -z "$RECHECK_BIN" ]; then
    for cand in "$(dirname -- "$BIN")/protlocus-recheck" "$HERE/../protlocus-recheck" protlocus-recheck; do
      command -v "$cand" >/dev/null 2>&1 && { RECHECK_BIN=$cand; break; }
    done
  fi
  [ -n "$RECHECK_BIN" ] && command -v "$RECHECK_BIN" >/dev/null || {
    echo "[FATAL] 找不到 protlocus-recheck（在项目根目录执行 make protlocus-recheck，或 -r 指定）" >&2; exit 1; }
else
  command -v hmmsearch >/dev/null || { echo "[FATAL] -R hmmer 需要 hmmsearch（HMMER3）" >&2; exit 1; }
  [ -s "$HERE/hmmrecheck.py" ] || { echo "[FATAL] 缺 $HERE/hmmrecheck.py" >&2; exit 1; }
fi
# Prefer GNU time from the active environment; BSD time does not accept -v.
TIME=()
TIME_PATH=$(type -P time || true)
if [ -n "$TIME_PATH" ] && "$TIME_PATH" --version 2>&1 | grep -q 'GNU'; then
  TIME=("$TIME_PATH" -v)
elif command -v gtime >/dev/null 2>&1; then
  TIME=(gtime -v)
fi

echo "[stage1] $BIN --prescreen --tier=$TIER --$MODE --trans -> $PREFIX.cand.paf" >&2
"${TIME[@]}" "$BIN" -I -u -t "$THREADS" --"$MODE" --prescreen --tier="$TIER" \
    --outs=0 -N 1000 --outn=1000 --acc-pri-ratio=0.1 --trans $EXTRA \
    "$GENOME" "$HMM" > "$PREFIX.cand.paf" 2> "$PREFIX.stage1.log" || {
  echo "[FATAL] protlocus 失败，见 $PREFIX.stage1.log" >&2; exit 1; }

if [ "$ENGINE" = native ]; then
  ZOPT=(); [ -n "$ZVAL" ] && ZOPT=(-Z "$ZVAL")
  echo "[stage2] $RECHECK_BIN -e $MAX_IE -> $PREFIX.final.paf" >&2
  "${TIME[@]}" "$RECHECK_BIN" -t "$THREADS" -e "$MAX_IE" "${ZOPT[@]}" \
      -T "$PREFIX.recheck.tsv" "$HMM" "$PREFIX.cand.paf" \
      > "$PREFIX.final.paf" 2> "$PREFIX.stage2.log" || {
    echo "[FATAL] protlocus-recheck 失败，见 $PREFIX.stage2.log" >&2; exit 1; }
else
  ZOPT=(); [ -n "$ZVAL" ] && ZOPT=(-Z "$ZVAL")
  echo "[stage2] hmmrecheck.py --max-ievalue $MAX_IE -> $PREFIX.final.paf" >&2
  "${TIME[@]}" python3 "$HERE/hmmrecheck.py" --paf "$PREFIX.cand.paf" --hmm "$HMM" \
      --out "$PREFIX.final.paf" --tsv "$PREFIX.recheck.tsv" \
      --max-ievalue "$MAX_IE" --threads "$THREADS" "${ZOPT[@]}" \
      2> "$PREFIX.stage2.log" || {
    echo "[FATAL] hmmrecheck 失败，见 $PREFIX.stage2.log" >&2; exit 1; }
fi

grep -h '\[recheck\]\|\[hmmrecheck\]' "$PREFIX.stage2.log" | tail -n 3 >&2 || true
if [ -n "$GENOME_FA" ]; then
  [ -s "$GENOME_FA" ] || { echo "[FATAL] -G 基因组 FASTA 不存在: $GENOME_FA" >&2; exit 1; }
  PYBIN=python3; command -v python3 >/dev/null || { echo "[FATAL] 第 3 段需要 python3" >&2; exit 1; }
  echo "[stage3] protlocus_genemodel.py -> $PREFIX.genes.gff3" >&2
  # shellcheck disable=SC2086
  $PYBIN "$HERE/protlocus_genemodel.py" -g "$GENOME_FA" -p "$PREFIX.final.paf" -E "$MAX_IE" -t "$THREADS" \
      -o "$PREFIX.genes.gff3" --faa "$PREFIX.genes.faa" $GM_EXTRA 2> "$PREFIX.stage3.log" || {
    echo "[FATAL] protlocus_genemodel 失败，见 $PREFIX.stage3.log" >&2; exit 1; }
  tail -n 1 "$PREFIX.stage3.log" >&2
fi
echo "[done] 最终命中: $PREFIX.final.paf  明细: $PREFIX.recheck.tsv" >&2
