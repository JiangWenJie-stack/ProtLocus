# ProtLocus 使用手册

源码项目版本：0.1.0　　上游基线：miniprot 0.18-r281　　许可证：MIT

ProtLocus 是一个把**蛋白质序列**或 **HMMER3 profile** 比对到基因组的剪接感知比对器（allowing splicing and frameshifts），基于 lh3/miniprot 开发。默认命令行行为与上游 miniprot 保持一致；在此之上 ProtLocus 提供以下可选能力：

| 能力 | 开关 | 用途 |
| --- | --- | --- |
| 高召回模式 | `--accuracy` / `--fast` / `--compact` | 重复序列多、旁系同源基因多的基因组上的基因发现；`--compact` 为紧凑基因组预设 |
| HMMER3 profile 查询 | 查询文件以 `.hmm` 结尾 | 用蛋白家族模型（单个或整库）在基因组中定位家族成员 |
| 内置预筛（整库 HMM） | `--prescreen --tier=...` | 数千到数万 profile 的全库扫描，在进程内生成候选位点，不依赖外部 mmseqs |
| GPU 后端 | `--gpu`（需 `protlocus`（以 `make gpu=1` 构建）） | 全基因组取种路径下大规模查询集的加速 |
| GPU 候选预筛 | `--gpu-prefilter`（需 `make cuda=1`） | 粗建链后用 GPU 给候选链打分排序，减少精修/比对阶段的候选数（见 10.7） |
| 小基因组召回预设（当前源码功能） | `--small-genome=auto\|on\|off` | `--prescreen` 下基因组 ≤ 50 Mb 时自动放开阈值，弱命中交复核裁决，召回 +4–5 pp（见 6.4b、9.4b） |
| 基因模型构建（当前源码功能） | `tools/protlocus_genemodel.py` / 流水线 `-G` | 把复核后的命中组装成 GFF3 基因模型 + 蛋白 FASTA（见 9.4c） |
| 候选 + HMM 复核流水线（5.0.0） | `tools/protlocus_hmm_pipeline.sh` / `hmmrecheck.py` | 整库扫描后把每条候选肽段交回其家族 HMM 重打分，最终命中带 hmmsearch 级 i-Evalue（见 9.4b） |

不使用上述任何开关时，输出与上游 miniprot 0.18 一致（本源码项目用内置 DPP3 样例做了逐字节校验）。


---

## 目录

1. [3.0.0 版本要点](#1-300-版本要点)
2. [发布包内容](#2-发布包内容)
3. [运行要求](#3-运行要求)
4. [安装与验证](#4-安装与验证)
5. [快速开始](#5-快速开始)
6. [命令行参考](#6-命令行参考)
7. [输入格式](#7-输入格式)
8. [输出格式](#8-输出格式)
9. [典型工作流](#9-典型工作流)
10. [GPU 使用指南](#10-gpu-使用指南)
11. [性能基准](#11-性能基准)
12. [`--profile` 统计字段](#12---profile-统计字段)
13. [Python 辅助工具](#13-python-辅助工具)
14. [常见问题](#14-常见问题)
15. [已知限制](#15-已知限制)
16. [从 1.0.0 / 2.0.0 升级](#16-从-100--200-升级)
17. [引用与许可](#17-引用与许可)

---

## 1. 3.0.0 版本要点

本节保留此前 3.0.0 开发阶段的功能记录和历史 benchmark，当前项目按独立公开版本 0.1.0 准备。相对 2.0.0 的变化全部集中在**整库 HMM 扫描**的候选位点注入模式；FASTA 与单 profile 查询的行为与 2.0.0 完全一致（完整列表见 `docs/NEWS.md`）：

- **`--prescreen`：内置预筛。** 基因组六框翻译为无终止子 ORF（≥ 30 aa，`--ps-min-orf`），按间隔 6-mer 建索引（`--ps-kmer`）；每个 profile 依次经过 MMseqs2 风格的 profile k-mer 预筛（`--ps-sens` 7.5，同一 ORF 对角线双命中，无 gap 得分 ≥ 15，每 profile 最多 `--ps-max-seqs` 4000 个 ORF）、带状 gap 比对（gap 11/1，`--ps-band` 32）与 Karlin-Altschul E 值。E 值低于 tier 阈值的位点像 `--cand-table` 一样注入精修/比对阶段；`--ps-dump=FILE` 可导出该表。酵母 × Pfam-A 38（56 线程，`--compact`，含约 10 s 预筛）：fast 36 s P0.988 R0.873；balanced 1.4 min P0.956 R0.923；sensitive 10.5 min P0.936 R0.965。与 `--cand-table` 互斥。
- **`--cand-table=FILE`：外部候选表。** TSV `query<TAB>chrom<TAB>start0<TAB>end0<TAB>strand[<TAB>evalue]` 取代全基因组取种/建链；未列出的查询不产生命中。表由 `tools/mmseqs2cand.py` 从 mmseqs2 profile-vs-ORF 搜索生成。
- **`--tier=fast|balanced|sensitive`** 与 **`--cand-max-evalue=FLOAT`**：按候选表第 6 列 E 值过滤。`--prescreen` 的 E 值自成尺度，tier 对应 E ≤ 0.1/100/10000（默认 balanced）；`--cand-table`（mmseqs E 值）对应 E ≤ 100/1000/10000。
- **`--acc-min-ps=NUM`（默认 60）**：HMM 命中的 profile Viterbi 分数下限，叠加在 DP 分数下限之上；`--acc-ps-rescue=AS[,POSR]` 为短模型的强比对提供豁免。对蛋白查询无影响。
- `--thr-table` / `--thr-fallback` / `--thr-use-ps`（按家族 decoy 阈值表）保留但不再是推荐协议：全局 AS ≥ 15 + ps ≥ 60 在所有实测工作点上不劣于 decoy 表。
- `--no-cand-diag`：精修链为空时不再用预筛的最佳无 gap PSSM 对角线取种（消融用）。
- Makefile：`hmm.o` 依赖头文件，避免选项结构体变更后陈旧目标文件导致候选表加载后崩溃。

### 1.1 2.0.0 版本要点（回顾）

相对 1.0.0 的主要变化：

- **HMMER3 profile 查询**。`.hmm` 文件可以含一个或任意多个 profile（例如 `cat *.hmm > library.hmm`），每个 profile 作为独立查询走与 FASTA 相同的多线程 mini-batch 流水线。种子来自 profile 一致序列及其"近平局"变体；候选区域用位置特异的 Viterbi DP 打分，分数写入 `AS:i`。`--gff`、`--aln`、`--trans`、`-u` 全部可用。
- **GPU 端延伸流水线**。`--accuracy` 模式约 74% 的 CPU 时间花在 16 位 x-drop 端延伸 DP 上；2.0.0 把这一步搬到 OpenCL 设备：输入准备在设备上完成，所有线程的请求汇入共享 arena，由常驻的多队列执行器（默认 8 个命令队列）分批推进，NVIDIA 设备上使用 warp 同步内核。结果与 CPU 内核逐位一致。
- **GPU 现在在大规模库扫描上快于 CPU**：5000 个 profile 时 120.6 s（GPU）对 153.8 s（CPU）；8.3 万 profile 全库在共享节点上 wall 2206 s 对 2296 s，CPU 时间 4.1 万 s 对 10.6 万 s（第 11 节）。1.0.0 中"GPU 慢于 CPU"的结论不再成立（≤1000 条查询时二者持平）。
- **修复一个上游 miniprot 也存在的潜在 bug**：CIGAR 回溯末端的 `D` 长度按核苷酸而非密码子计数，导致以 deletion 开头的子比对 reference 跨度被重复计算，触发 `mp_extra_cal` 断言。实践中只有 GPU 的等分路径会产出这类比对，所以在全库扫描时才暴露。
- 修复 1.0.0 中 OpenCL HMM 内核少设一个 `__local` 参数、静默回退 CPU 的问题。
- 新增 `make static=1`（全静态 CPU 可执行文件）；修复不带 `gpu=1` 时 `nasw-ext-gpu.c` 编译失败。
- 新增一组诊断环境变量（第 10.5 节）。

---

## 2. 源码项目内容

源码和 Makefile 位于项目根目录；`tools/` 为 Python 工具和流水线，`test/` 为测试与小样例，`docs/` 为手册和技术资料，`packaging/` 为尚未提交的 Conda 配方草稿。项目不附带预编译程序或大型数据库。

本文的所有命令均在项目根目录运行。当前版本标识为 `0.1.0`；历史版本的基准与回顾仅供参考，不代表本次重新执行的结果。

---

## 3. 运行要求

### 3.1 `protlocus`（CPU 版）

- Linux x86_64，CPU 支持 SSE2；检测到 AVX2 时自动启用 256 位端延伸 DP。
- 从源码编译，需要 C99 编译器、GNU Make 和 zlib 开发库。CPU 版不需要 CUDA 或 OpenCL；Python 工具需要 Python 3。静态构建需额外提供静态运行库。
- 内存：基因组索引约为基因组大小的数倍（小鼠 GRCm39 索引载入后约 3–4 GB）；56 线程全库扫描峰值约 31 GB（第 11 节）。

### 3.2 `protlocus`（以 `make gpu=1` 构建）（OpenCL 版）

- 除上述要求外，需要系统 `libz.so.1`、`libOpenCL.so.1`（ICD loader）以及一个 GPU 厂商 ICD（NVIDIA 驱动自带）。
- 已在 NVIDIA L20（OpenCL 3.0，驱动自带 ICD）上测试；不支持 `cl_khr_subgroups` 的设备也可运行（自动使用可移植内核）。
- 不传 `--gpu` 时行为与 `protlocus` 完全一致。

### 3.3 源码编译

C99 编译器、GNU Make、pthread、libm、zlib 开发包；`make check` 另需 Python 3。GPU 版链接时需要 `libOpenCL.so`（不需要 OpenCL 头文件，源码自带最小声明）。

---

## 4. 安装与验证

### 4.1 构建与测试

```bash
make -j2
make check
```

当前项目不提供旧二进制发布包的 `verify.sh`、校验清单或构建记录。GitHub 自动构建配置见 `.github/workflows/ci.yml`。

### 4.2 手动检查

```bash
./protlocus --version  # 0.1.0
./protlocus-recheck -v  # 0.1.0
./protlocus --help
man ./protlocus.1
```

### 4.3 从源码编译

```bash
make clean
make -j                # 动态链接 CPU 版
make clean && make -j static=1       # 全静态 CPU 版（需要 libz.a / libc.a）
make clean && make -j gpu=1          # OpenCL 版（推荐的 GPU 构建）
make gpu=1 check       # 有 GPU 时运行 GPU-vs-CPU 测试
make clean && make -j cuda=1 NVCC=/path/to/nvcc CUDA_HOME=/path/to/cuda   # CUDA 版：仅 gap DP，不含端延伸流水线与 HMM GPU 打分
```

注意事项：

- `static=1` 与 `gpu=1`/`cuda=1` 不能同时使用（ICD loader 必须动态链接）。
- 用 Conda 工具链编译 GPU 版时，链接到的可能是 Conda 的 `libOpenCL.so`，它没有配置 ICD，运行时会报 `GPU unavailable (clGetPlatformIDs failed with OpenCL status -1001)` 并回退 CPU。解决：用系统 gcc/ld 编译（`make gpu=1 CC=/usr/bin/gcc`），或设置 `OCL_ICD_VENDORS` 指向厂商 ICD 目录。
- 旧版 gcc（如 CentOS 7 的 4.8.5）编译出的程序在某些序列上可能非常慢，建议 gcc 8 以上。

---

## 5. 快速开始

```bash
# 1. 建索引（同一基因组多次使用时只建一次）
./protlocus -t 16 -d genome.mpi genome.fa        # genome.fa.gz 也可

# 2. 普通模式：与上游 miniprot 一致
./protlocus -I -t 16 genome.mpi proteins.fa > aln.paf

# 3. 高召回模式 + GFF3
./protlocus -I -u -t 16 --accuracy --gff genome.mpi proteins.fa > pred.gff 2> run.log

# 4. HMMER3 profile 家族搜索（单个/少量 profile，全基因组取种）
./protlocus -I -t 16 --accuracy genome.mpi family.hmm > family.paf

# 5. 整库 HMM 扫描：内置预筛（3.0.0 推荐；紧凑基因组如酵母用 --compact 代替 --accuracy）
cat pfam/*.hmm > library.hmm
./protlocus -I -u -t 56 --accuracy --prescreen --tier=balanced genome.mpi library.hmm > library.paf 2> library.log

# 6. 整库 HMM 扫描：全基因组取种 + GPU（2.0.0 路径，不用预筛）
./protlocus -I -u -t 56 --accuracy --gpu genome.mpi library.hmm > library.paf 2> library.log
```

说明：`-I` 按基因组大小自动设置最大内含子长度（min(max(3.6·√refLen, 10 k), 300 k)），不加则默认 200 k；`-u` 让未比对上的查询也输出一行，便于统计召回。程序日志全部写到标准错误。

---

## 6. 命令行参考

```
protlocus [options] <ref.fa|ref.mpi> <query.faa|query.hmm> [more query files...]
```

多个查询文件按顺序处理，FASTA 与 `.hmm` 可以混用。数值参数可带 `k`/`m`/`g` 后缀（如 `-c 50k`、`--gpu-min-cells=1m`）。

### 6.1 索引参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-k INT` | 6 | 全基因组索引的 k-mer 长度（氨基酸） |
| `-M INT` | 1 | modimizer 位数，采样率 1/2^M；增大可降低内存但降低敏感度 |
| `-L INT` | 30 | 建索引的最短 ORF 长度（aa） |
| `-T INT` | 1 | NCBI 翻译表（1–33，7–8、17–20 除外）。索引参数，换表需重建索引 |
| `-b INT` | 8 | 基因组分块大小 2^b bp |
| `-d FILE` | | 把索引写入 FILE（`.mpi`） |

### 6.2 建链参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-S` | | 关闭剪接，等价于 `-G1k -J1k -e1k` |
| `-c NUM` | 50k | 出现次数 ≥ NUM 的 k-mer 忽略 |
| `-G NUM` | 200k | 最大内含子长度，覆盖 `-I` |
| `-I` | | 按基因组大小自动设置最大内含子长度 |
| `-w FLOAT` | 0.75 | 建链时对数 gap 罚分权重 |
| `-n NUM` | 10 | 一条链最少 syncmer 数 |
| `-m NUM` | 0 | 最小链得分 |
| `-l INT` | 5 | 第二轮建链 k-mer 长度 |
| `-e NUM` | 10k | 链两端向外延伸的最大距离 |
| `-p FLOAT` | 0.5 | 次优链/比对得分低于最优的 FLOAT 倍时丢弃 |
| `-N NUM` | 30 | 最多保留 NUM 条次优链/比对 |
| `--max-skip NUM` | | 建链跳跃上限（调试） |
| `--no-pre-chain` | | 关闭预建链（调试） |

### 6.3 比对参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-O INT` | 11 | gap open |
| `-E INT` | 1 | gap extension；长度 g 的 gap 代价 O + g·E |
| `-J INT` | 29 | 内含子 open 罚分 |
| `--J2 INT` | | 端延伸阶段的内含子 open 罚分 |
| `-F INT` | 23 | 移码或读框内终止密码子罚分 |
| `-C FLOAT` | 1 | 剪接信号罚分权重，0 = 忽略剪接信号 |
| `-B INT` | 5 | 比对到达蛋白末端的奖励分 |
| `-j INT` | 1 | 剪接模型：2 = 脊椎动物/昆虫，1 = 通用，0 = 无 |
| `--spsc FILE` | | 外部剪接位点得分文件（格式见 7.4）；未显式设置时自动把 `-J`、`--J2` 提高 10 |
| `--spsc0 INT` | -7 | `--spsc` 文件中未列出的位点得分 |
| `--spsc-max INT` | 14 | 剪接得分上限 |
| `--xdrop INT` | 100 | x-drop 阈值 |
| `--ie-coef FLOAT` | 0.5 | 对数内含子长度罚分系数（实验性） |

### 6.4 高召回模式参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--accuracy` | | 启用全部高召回机制：重复种子救回、按 contig/链隔离建链、放宽候选保留、无可信锚点全窗口 DP 救回、可疑剪接点重精修、复合重排序 |
| `--fast` | | `--accuracy` 的速度预设，等价于 `--accuracy --acc-end-dyn --acc-end-aa-cap`；比对阶段快约 20–25%，少量召回代价 |
| `--profile` | | 运行结束时向 stderr 输出一行 `[M::profile]` 阶段耗时与工作量统计（第 12 节） |
| `--acc-repeat-cap=NUM` | 64 | 每个被过滤的高频 k-mer 最多采样 NUM 个锚点 |
| `--acc-anchor-budget=NUM` | 250000 | 每条查询额外救回锚点上限 |
| `--acc-rescue-cells=NUM` | 5e7 | 无锚点全窗口 DP 每候选的 cell 上限 |
| `--acc-pri-ratio=FLOAT` | 0.35 | 放宽后的候选得分比 |
| `--acc-best-n=NUM` | 100 | 放宽后的次优候选数 |
| `--acc-microexon=NUM` | 30 | 对 ≤ NUM aa 的查询插入尝试微外显子重比对 |
| `--acc-rerank-margin=NUM` | 10 | 复合得分相对原始 dp_max 的加分上限 |
| `--acc-end-ext=NUM` | 10000 | 端延伸初始窗口（nt） |
| `--acc-end-retry=NUM` | 89 | 窗口受限查询未比对上时的重试预算 |
| `--acc-end-retry-win=NUM` | 10000 | 第二次尝试（短外显子 DP）的 nt 窗口 |
| `--acc-end-ext-retry` | | 对救回候选也用 io_end 重试端延伸（更慢） |
| `--acc-end-aa-cap` / `--acc-end-aa-ext=NUM` | / 100000 | 限制救回候选首次端延伸的 aa 侧范围 |
| `--acc-end-dyn` | | 端延伸 nt/aa 上限随查询长度缩放 |
| `--acc-end-ext-min`, `--acc-end-nt-per-aa`, `--acc-end-aa-min`, `--acc-end-aa-ratio` | 300 / 15 / 50 / 0.5 | `--acc-end-dyn` 的下限与系数 |
| `--acc-cov-bonus`, `--acc-splice-bonus`, `--acc-noncan-penalty`, `--acc-fs-penalty`, `--acc-stop-penalty`, `--acc-end-bonus` | | 复合重排序的各项加/减分 |
| `--acc-no-repeat`, `--acc-no-vid-chain`, `--acc-no-rescue`, `--acc-no-rerank`, `--acc-no-junc-refine`, `--acc-no-end-ext-cap` | | 单独关闭某一机制（消融实验） |

各机制的算法说明与消融数据见 `docs/ACCURACY.md`。

`--compact`：面向紧凑、基因密集、内含子少的基因组（如酵母）的识别预设，隐含 `--accuracy`：提高锚点救回预算，开启 `--acc-strict-multi-exon` 与 `--acc-strict-span-ratio=4` 拒绝跨入相邻基因的命中，设 `--acc-min-score=15 --acc-min-pos-ratio=0.50` 压制基因组中不存在的家族的垃圾最佳命中，并按基因组大小缩放最大内含子（同 `-I`）。

### 6.4b 候选位点注入与内置预筛（HMM 整库扫描，3.0.0）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--prescreen` | | 在进程内构建基因组六框 ORF 间隔 k-mer 索引，为每个 profile 生成候选位点（profile k-mer 预筛 → 无 gap 对角线 → 带状 gap 比对 → E 值），取代全基因组取种/建链。与 `--cand-table` 互斥；未给 `--tier`/`--cand-max-evalue` 时默认 balanced |
| `--tier=STR` | | `fast` / `balanced` / `sensitive`：候选位点 E 值上限。`--prescreen` 下为 E ≤ 0.1 / 100 / 10000；`--cand-table` 下为 E ≤ 100 / 1000 / 10000；`sensitive` 还把预筛 k-mer 阈值降 1 |
| `--cand-max-evalue=FLOAT` | 0 | 任意 E 值上限（给了 `--tier` 时忽略）；≤ 0 保留全部位点 |
| `--cand-table=FILE` | | 外部候选表 TSV：`query<TAB>chrom<TAB>start0<TAB>end0<TAB>strand[<TAB>evalue]`（`tools/mmseqs2cand.py` 生成）；只精修/比对列出的位点，未列出的查询无命中 |
| `--ps-sens=FLOAT` | 7.5 | 预筛敏感度（mmseqs `-s` 尺度），决定 profile k-mer 得分阈值 |
| `--ps-kmer=INT` | 6 | 间隔 k-mer 长度 5–7（索引 20^k 桶：k=6 约 0.5 GB，k=7 约 10 GB） |
| `--small-genome=STR` | auto | `auto`/`on`/`off`（当前源码功能）：与 `--prescreen` 联用，基因组总长 ≤ `--small-genome-size`（或 `on`）时切换到小基因组召回预设 `--ps-kmer=5 --acc-min-ps=0 --acc-min-score=0 --acc-min-pos-ratio=0`；命令行显式给出的这四项保持不变；弱命中交 `protlocus-recheck`（E≤1e-5）裁决。`off` 与 5.0.0 输出逐字节一致；无 `--prescreen` 时 `on` 仅告警 |
| `--small-genome-size=NUM` | 50M | 触发 `--small-genome=auto` 的基因组总长阈值 |
| `--ps-kmer-thr=NUM` | 0 | 显式 k-mer 得分阈值（half-bits），覆盖 `--ps-sens`；≤ 0 由 `--ps-sens` 推导（k=6：22） |
| `--ps-kmer-max=NUM` | | 每个 profile 窗口生成的相似 k-mer 上限 |
| `--ps-min-orf=INT` | 30 | 忽略短于 INT aa 的无终止子 ORF（同 esl-translate `-l`） |
| `--ps-min-ungapped=NUM` | 15 | 无 gap 对角线得分下限（half-bits） |
| `--ps-max-seqs=NUM` | 4000 | 每个 profile 进入 gap 比对阶段的 ORF 数 |
| `--ps-band=INT` | 32 | gap 比对带宽（aa）；≤ 0 全矩阵 |
| `--ps-dump=FILE` | | 同时写出生成的候选表（与 `--cand-table` 同格式，第 6 列 E 值） |
| `--no-cand-diag` | | 精修 5-mer 链为空时不用预筛的最佳无 gap PSSM 对角线取种（消融用；关闭会丢失分化大的家族成员） |
| `--acc-min-ps=NUM` | 60 | HMM 命中的 profile Viterbi 分数（PAF `ps`）下限，叠加在 AS 下限之上；≤ 0 关闭；对蛋白查询无效 |
| `--acc-ps-rescue=AS[,POSR]` | | AS ≥ AS 且正分比 ≥ POSR 的命中免于 `--acc-min-ps`（短模型的强比对 ps 可能不到 60）；AS ≤ 0 关闭 |
| `--thr-table=FILE` | | 按查询的 DP 得分下限表（`query<TAB>min_AS`，如 decoy 校准的按家族阈值），对列出的查询取代 `--acc-min-score`。不再是推荐协议 |
| `--thr-fallback=NUM` | 0 | 表中未列出的查询的下限；≤ 0 保持 `--acc-min-score` |
| `--thr-use-ps` | | 把上述下限作用于 `ps` 而非 `AS`（不推荐，用 `--acc-min-ps`） |

HMM 取种参数（全基因组取种路径）：`--hmm-seed-max`（粗阶段每 k 窗口的 profile 衍生种子数，0 = 仅一致序列）、`--hmm-seed-max2`（精修阶段同上）、`--hmm-seed-loss`、`--hmm-seed-pos-loss`（发射损失预算）。

### 6.5 GPU 参数（仅 `protlocus`（以 `make gpu=1` 构建））

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--gpu` | | 在 OpenCL 设备上执行端延伸 DP（全部请求）与大 gap DP；设 `MP_HMM_GPU=1` 时再加 HMM profile 打分。设备不可用时打印警告并回退 CPU |
| `--gpu-device=INT` | 0 | OpenCL GPU 序号（从 0 开始） |
| `--gpu-min-cells=NUM` | 1M | 锚点间 gap DP 提交 GPU 的最小 cell 数（nt 长 × aa 长）；不影响端延伸流水线 |
| `--gpu-prefilter` | | 粗建链后用批处理 GPU 无 gap X-drop 打分给候选链排序，只保留前 K 条（见 10.7）；仅 CUDA 构建（`make cuda=1`）支持，其他构建下会打印警告并忽略该参数 |
| `--gpu-prefilter-topk=NUM` | 32 | `--gpu-prefilter` 每条查询保留的候选链数 K |
| `--gpu-prefilter-diversity=NUM` | 1 | 每个目标序列/链方向分组至少保留的候选链数 |
| `--gpu-prefilter-xdrop=NUM` | 40 | GPU 无 gap 延伸打分的 X-drop 阈值 |
| `--gpu-prefilter-adaptive` | | 不用固定 K，按最优与次优候选链的分差在 {1,2,4,8,`--gpu-prefilter-topk`} 中自适应选 K；隐含 `--gpu-prefilter` |

### 6.6 输入输出参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-t INT` | 4 | 线程数 |
| `--gff` | | 输出 GFF3，同时保留 `##PAF` 行 |
| `--gff-only` | | 仅输出 GFF3 |
| `--gtf` | | 输出 GTF（不含详细比对） |
| `--aln` | | 输出 `##ATN`/`##ATA`/`##AQA` 三行残基级比对 |
| `--trans` | | 输出翻译后的蛋白序列（`##STA` 行） |
| `--no-cs` | | 不输出 `cs` 标签 |
| `--max-intron-out NUM` | 200 | `--aln` 中长于 NUM 的内含子只显示两端各 NUM/2 bp |
| `-P STR` | MP | GFF3/GTF ID 前缀 |
| `--gff-delim CHAR` | | GFF3 ID 改为 `QueryName<CHAR>HitIndex` |
| `-u` | | 未比对上的查询也输出一行（`name len 0 0 * * 0 0 0 0 0 0`） |
| `--outn NUM` | 1000 | 每查询最多输出 min{NUM, -N} 条比对 |
| `--outs FLOAT` | 0.99 | 仅输出得分 ≥ FLOAT × 最优分的比对 |
| `--outc FLOAT` | 0.1 | 仅输出覆盖 ≥ FLOAT 的查询比例的比对 |
| `-K NUM` | 2M | 每个 mini-batch 的查询总长度（aa） |
| `--help` / `--version` | | 帮助 / 版本 |

---

## 7. 输入格式

### 7.1 基因组

FASTA 或 gzip FASTA，或事先由 `-d` 生成的 `.mpi` 索引。索引与翻译表 `-T` 绑定；ProtLocus 拒绝加载旧版 miniprot 或不同 `-T` 生成的索引。超过 2 Gbp 的染色体受支持。

### 7.2 蛋白质 FASTA

标准氨基酸 FASTA / gzip FASTA。序列名取 `>` 后第一个空白前的字段。多文件按顺序处理。

### 7.3 HMMER3 profile（`.hmm`）

文件名必须以小写 `.hmm` 结尾，内容为 HMMER3 文本格式（`hmmbuild` 输出，HMMER3/f 已测试）。每个 profile 需要：

| 字段 | 用途 |
| --- | --- |
| `NAME` | PAF 第 1 列查询名 |
| `LENG` | PAF 第 2 列查询长度（match 状态数，单位 aa） |
| `ALPH amino` | 只支持氨基酸字母表 |
| `HMM` 表头行 | 20 个氨基酸列顺序 |
| `COMPO` 行 | 可有可无，会被跳过 |
| 每个位置的 match 发射（20 个）+ insert 发射 + 7 个转移 | 打分用 |
| `//` | 每个 profile 的结束标记，**必须存在** |

一个文件可包含任意多个 profile：

```bash
cat family1.hmm family2.hmm > two.hmm
find pfam_dir -name '*.hmm' -print0 | xargs -0 cat > library.hmm
```

处理方式：

1. **解析**流式进行，每个 profile 在读入后立即进入 mini-batch，内存不随库大小线性增长（8.3 万 profile、15 GB 的库在 56 线程下峰值 RSS 约 33 GB，与 FASTA 输入同量级）。
2. **种子**：以每列发射最高的残基作为一致序列，与蛋白查询一样做 syncmer 采样；另外对第二高发射与最高发射相差 ≤ 6 分的列生成替代残基，组合成最多 64 条变体序列一起取种，避免歧义列丢失种子。
3. **打分**：候选区域翻译后用 profile 的 match/insert/delete 状态做位置特异的 Viterbi DP；一条查询的所有候选区域在一次批处理中打分（默认 CPU；`MP_HMM_GPU=1` 时 ≤1023 个 match 状态的 profile 在 OpenCL 上每区域一个 work-group，更长的回退 CPU）。Viterbi 分数写入 `AS:i`，并用于候选排序。
4. **回溯**：残基比对与 CIGAR 使用一致序列，因此 `--gff`、`--aln`、`--trans`、`-u` 与蛋白查询完全一样。

> 提示：profile 打分只作用于已经通过种子/建链找到的候选区域。如果家族分化极大、一致序列在基因组中找不到 k-mer 种子，该家族仍会漏检；请先用有真值的家族评估召回。`--accuracy` 会显著提高 HMM 查询的召回（重复种子救回与放宽候选保留对 profile 尤其有效），推荐与 `.hmm` 一起使用。

### 7.4 剪接位点得分文件（`--spsc`）

TAB 分隔，每行 `contig  offset  +|-  D|A  score`。offset 为剪接点在正链上的 0-based 位置，`D`/`A` 表示供体/受体，score 为整数（可正可负）。可由 minisplice 直接生成，或用 `tools/junctions_to_spsc.py` 从 STAR `SJ.out.tab` / BED 等 RNA-seq 剪接证据转换（第 13 节）。

---

## 8. 输出格式

### 8.1 PAF（默认）

TAB 分隔，前 12 列：

| 列 | 类型 | 含义 |
| --- | --- | --- |
| 1 | str | 查询名（蛋白名或 profile `NAME`） |
| 2 | int | 查询长度（aa；profile 为 `LENG`） |
| 3 | int | 查询起点（0-based） |
| 4 | int | 查询终点（0-based，开区间） |
| 5 | char | `+` 正链 / `-` 负链 |
| 6 | str | contig 名 |
| 7 | int | contig 长度 |
| 8 | int | contig 起点（原始链坐标） |
| 9 | int | contig 终点 |
| 10 | int | 匹配的核苷酸数 |
| 11 | int | 比对中的核苷酸数（不含内含子） |
| 12 | int | 比对质量（0–255，255 表示缺失） |

可选标签：

| 标签 | 类型 | 含义 |
| --- | --- | --- |
| `AS` | i | DP 比对分；profile 查询时为 Viterbi 分 |
| `ms` | i | 不含内含子的比对分 |
| `np` | i | 得分为正的氨基酸匹配数 |
| `fs` | i | 移码数 |
| `st` | i | 读框内终止密码子数 |
| `da` | i | 到最近起始密码子的距离 |
| `do` | i | 到最近终止密码子的距离 |
| `cg` | Z | 蛋白 CIGAR |
| `cs` | Z | 差异字符串 |

蛋白 CIGAR 操作符：

| Op | 消耗 |
| --- | --- |
| `nM` | 3n nt + n aa（匹配） |
| `nI` | n aa（插入） |
| `nD` | 3n nt（缺失，**n 为密码子数**） |
| `nF` | n nt（移码缺失） |
| `nG` | n nt + 1 aa（移码匹配） |
| `nN` | n nt（0 相位内含子） |
| `nU` | n nt + 1 aa（1 相位内含子） |
| `nV` | n nt + 1 aa（2 相位内含子） |

`cs` 标签：`:N` 连续 N 个相同氨基酸；`*acgX` 替换（ref 密码子→query 氨基酸）；`+ABC` 相对 ref 插入的氨基酸；`-acg` 从 ref 删除的核苷酸；`~gt123ag` 内含子长度与剪接信号。

用 `-u` 时未比对上的查询输出一行 `name  len  0  0  *  *  0  0  0  0  0  0`，方便统计。

### 8.2 GFF3（`--gff` / `--gff-only`）

特征类型 `mRNA`、`CDS`、`stop_codon`（仅当比对到达蛋白 C 端且下一个密码子是终止子；按 GenCode 规则 stop_codon 不计入 CDS）。属性：`ID`、`Parent`、`Rank`、`Identity`、`Positive`、`Donor`（非 GT 时）、`Acceptor`（非 AG 时）、`Frameshift`、`StopCodon`、`Target`。`--gff` 在 GFF3 中插入 `##PAF` 行保留完整比对信息，`--gff-only` 不输出这些行。

### 8.3 GTF（`--gtf`）

基本 GTF，不含残基级信息。

### 8.4 残基级比对（`--aln`）

每条比对后输出三行：`##ATN` 目标核苷酸、`##ATA` 翻译氨基酸（`!` 表示移码插入，对应 `F`；`$` 表示移码替换，对应 `G`）、`##AQA` 查询氨基酸（profile 查询时为一致序列）。长内含子按 `--max-intron-out` 折叠为 `~LEN~`。

### 8.5 日志（stderr）

`[M::...]` 行报告索引统计、每个 mini-batch 的进度（`mapped N sequences`）、版本、命令行、`Real time / CPU / Peak RSS`。`--gpu` 时首行有 `[M::gpu] using device N: <name>`。`[WARNING]` 行值得关注（例如 `--spsc` 校验不一致、GPU 不可用）。

---

## 9. 典型工作流

### 9.1 新物种基因注释（蛋白证据）

```bash
./protlocus -t 32 -d genome.mpi genome.fa.gz
./protlocus -I -u -t 32 --accuracy --gff genome.mpi uniprot_sprot.fa.gz > sprot.gff 2> sprot.log
python3 tools/filter_paf.py --min-cov 0.5 <(grep '^##PAF' sprot.gff | cut -f2-) > sprot.cov50.paf
```

- 召回优先用 `--accuracy`；接受少量结果变化换 ~20% 速度用 `--fast`；需要与上游 miniprot 逐字节一致时都不加。
- 四基因组（小鼠/大鼠/蛙/斑马鱼，1114 蛋白，56 线程）基准中 `--accuracy` 默认配置与"不限预算穷举"召回相同，耗时 65.2 s 对 158.3 s。

### 9.2 用 RNA-seq 剪接证据约束比对

```bash
python3 tools/junctions_to_spsc.py --format star SJ.out.tab > splice.spsc
./protlocus -I -t 32 --accuracy -j 2 --spsc=splice.spsc genome.mpi proteins.fa > aln.paf
```

### 9.3 蛋白家族定位（单个或少量 HMM）

```bash
hmmbuild IL2.hmm IL2_aligned.sto          # 或直接使用 Pfam 等已有 profile
./protlocus -I -u -t 16 --accuracy --gff genome.mpi IL2.hmm > IL2.gff
```

用 `-u` 可以立刻看到哪些 profile 完全没有命中。同一 profile 的多个位点按 `AS` 排序，`--outn`/`--outs` 控制输出数量。

### 9.4 全库扫描（数千到数万 profile）

**推荐（3.0.0）：内置预筛。** 不需要外部 mmseqs，也不需要 GPU：

```bash
cat pfam/*.hmm > Pfam-A.hmm
./protlocus -I -u -t 56 --compact --prescreen --tier=balanced --ps-dump=cand.tsv \
    genome.mpi Pfam-A.hmm > library.paf 2> library.log       # 紧凑基因组；大基因组用 --accuracy
```

- `--tier=fast` 用于快速普查，`sensitive` 用于尽可能高的召回；三档在酵母 × Pfam-A 38 上的耗时/P/R 见第 1 节。
- `--ps-dump` 导出的候选表可直接作为 `--cand-table` 重用（搭配 `--cand-max-evalue` 在不重跑预筛的前提下扫描多个工作点）。
- 预筛索引与基因组大小成正比；`--ps-kmer=7` 约 10 GB，仅在内存充足时使用。

**外部候选表（mmseqs2）：**

```bash
python3 tools/mmseqs2cand.py --help      # 从 profile-vs-ORF mmseqs2 结果生成 cand.tsv
./protlocus -I -u -t 56 --compact --cand-table=cand.tsv --tier=balanced genome.mpi Pfam-A.hmm > library.paf
```

**全基因组取种 + GPU（2.0.0 路径）：** 适用于不使用候选表、完全依赖 k-mer 取种的扫描。

```bash
NS_EXT_GPU_DEBUG=1 ./protlocus -I -u -t 56 --accuracy --gpu \
    genome.mpi library.hmm > library.paf 2> library.log
grep -E 'ns_ext_gpu|Real time|Peak RSS' library.log
```

- 库越大 GPU 收益越明显（第 11 节）。线程数取物理核数；GPU 版在 56 线程时设备成为瓶颈，CPU 空闲会自然出现。
- 输出按 mini-batch 顺序，与输入 profile 顺序一致；同一库重跑结果逐字节一致。
- 拼接库时保证每个 profile 以 `//` 结束；损坏的 profile 会导致解析在该处停止并报错。

### 9.4b 全库扫描 + HMM 复核（5.0.0 推荐的最终口径）

`--prescreen` 的候选命中以速度和灵敏度见长，但 hit 级假发现率随基因组重复含量上升
（七基因组 decoy 实验：balanced 档 16%–77%）。5.0.0 的复核段用**原生 C 复核器
`protlocus-recheck`** 把每条候选肽段交回它**自己家族的 profile** 重打分，给最终命
中统计校准的显著性，全程不依赖 HMMER：

```bash
make protlocus-recheck    # 需要 C 编译器及 zlib；预编译版本的要求见对应包说明

# 一键（第 1 段 protlocus + 第 2 段原生复核）
bash tools/protlocus_hmm_pipeline.sh -g genome.mpi -m Pfam-A.hmm -o out \
    -t 56 -T balanced -M accuracy            # 紧凑基因组 -M compact
# 产物：out.final.paf（最终命中，带 fB:f:/fE:f:）、out.recheck.tsv（逐候选明细）、
#       out.cand.paf（第 1 段候选）、out.stage{1,2}.log（/usr/bin/time -v 资源日志）

# 分步等价写法
./protlocus -I -u -t 56 --accuracy --prescreen --tier=balanced \
    --outs=0 -N 1000 --outn=1000 --acc-pri-ratio=0.1 --trans \
    genome.mpi Pfam-A.hmm > out.cand.paf
./protlocus-recheck -t 56 -e 1e-5 -T out.recheck.tsv \
    Pfam-A.hmm out.cand.paf > out.final.paf
```

原理与代价：

- `--trans` 让每条 PAF 命中后多输出一行 `##STA\t<肽段>`——沿比对路径（跨内含子、
  跳过移码）翻译的基因组肽段。跨内含子的域在六框单帧 ORF 里是断裂的，在这里是
  完整的，所以复核不损失剪接域召回。
- `protlocus-recheck` 是 plan-7 统计模型（Eddy, PLoS Comput Biol 2011）的独立
  C 实现：multihit local 配置（占用率加权入口、目标长度 N/C/J 模型）、log 空间
  完整 Forward、null1 几何长度模型、**后验加权 null2 组成偏倚校正**（ByExpectation
  方法，经完整 Backward 后验计算；`-n` 可关闭）；E 值用 profile 自带的
  `STATS LOCAL FORWARD tau lambda` 校准行换算（指数尾 P 值 × Z）。与 hmmsearch
  报告值的差异：这里把整条候选肽段当一个 envelope 做全序列打分与 null2（非逐
  domain envelope）——decoy 实验是最终的校准核验。
- 复核是**成对打分**而非交叉积：每条肽段只对一个模型算 Forward。相对「全部
  profile × 全部六框 ORF」的 hmm6f，计算量约降 1/N_profile（Pfam-A 约两万分之
  一），再乘以候选肽段远小于六框 ORF 的比例——复核段通常只需分钟级。
- 过滤口径：Forward E 值 ≤ `-e`（默认 1e-5）。`-Z` 默认取唯一候选肽段数；要与
  hmm6f 严格同刻度，传该物种六框 ORF 数（更保守）。
- `out.recheck.tsv` 每候选一行：肽长、读框内终止子数（`n_stop`，假基因信号）、
  Forward bits / E 值、是否通过。`-a` 时 PAF 保留落选行并追加 `hp:i:0`。
- **交叉验证**：`-R hmmer` 把第 2 段换成 `tools/hmmrecheck.py`（逐家族调
  HMMER3 的 hmmfetch/hmmsearch，按 best domain i-Evalue≤1e-5 过滤，输出
  `hB:f:/hE:f:` tag）——用来核对原生引擎与 hmmsearch 的一致性。
- 当前源码：基因组 ≤ 50 Mb 时 `--prescreen` 自动启用小基因组召回预设（`--small-genome`，见 6.4b），第 1 段放开 `--acc-min-*` 与 k=5，弱命中全部交由本段（E≤1e-5）裁决；不要再额外放宽 sensitive 档预滤（候选爆炸会抬高 Z、反而降低召回）。
- 注意：复核只会收紧（过滤）第 1 段的候选；要提高召回上限，把第 1 段换
  `--tier=sensitive` 再复核。

### 9.4c 由命中构建基因模型（第 3 段）

`tools/protlocus_genemodel.py`（纯标准库 Python ≥ 3.6）把第 2 段的 `final.paf` 转成 GFF3（`gene/mRNA/exon/CDS`）和蛋白 FASTA，只用基因组与库命中：

```bash
# 一键：在 9.4b 的流水线上加 -G（基因组 FASTA，不是 .mpi）
bash tools/protlocus_hmm_pipeline.sh -g genome.mpi -m Pfam-A.hmm -o out -t 56 -G genome.fa
#   产物另有 out.genes.gff3、out.genes.faa、out.stage3.log
bash tools/protlocus_hmm_pipeline.sh ... -G genome.fa -X "--no-introns"   # 原核

# 分步
python3 tools/protlocus_genemodel.py -g genome.fa -p out.final.paf -E 1e-5 -t 16 \
    -o out.genes.gff3 --faa out.genes.faa
# 评估（RefSeq/TAIR 等参考 GFF3；--paf 额外给出"被命中覆盖基因"上的召回）
python3 tools/evaluate_genemodel.py --ref ref.gff3 --pred out.genes.gff3 --paf out.final.paf
```

要点：

- 命中 → 带密码子相位的外显子块（同链重叠命中框/结构一致则合并，否则留显著性高者；含移码/框内终止子的命中默认跳过，`--allow-fs` 可保留）。
- 相邻证据单元先尝试同框直接融合（中间无终止子），否则在 GT/GC..AG 内含子上做保持读框的 DP（剪接位点 PWM、密码子/二联密码子对数比、内含子长度先验），须优于「两个独立基因」基线 `--link-min` 才连接。
- 两端按 ORF 延伸：3' 到首个同框终止子，5' 到累计编码得分最佳的 ATG（`--start-slack`）；基因组内含子足够多时可启用一内含子的剪接端延伸（自动）。
- 全部模型参数由基因组自身命中自训练（剪接位点 PWM 从通用先验起步并 EM 精修）。紧凑基因组的比对内含子多为假象（酵母 1280 个命中内含子仅 25 个与注释一致），可信种子不足时判为「无可训练内含子」：通用先验、最大内含子 2 kb、丢弃含无支持内含子的命中、关闭剪接端延伸。原核用 `--no-introns`（可选 `--start-codons ATG,GTG,TTG`）。
- 实测（Pfam-A 命中，E≤1e-5，对 RefSeq CDS，默认参数）：酵母 nt 灵敏度 0.84 / 特异度 0.99，87.7% 预测基因 CDS 链与参考完全一致，69% 的参考基因被精确恢复；大肠杆菌（`--no-introns`）0.89 / 0.98，65% 完全一致；拟南芥（内含子丰富）0.55 / 0.81，仅约 22% 完全一致——多数 CDS 在 Pfam 域之外，缺失的 5'/3' 与域间外显子是主要误差。没有任何 Pfam 命中的基因找不到。内含子丰富的基因组请把输出当作「以蛋白域为锚的基因骨架 / 提示」（如作为 Augustus 的 hints，或与 ab initio 模型并用），不是最终模型。

### 9.5 关闭剪接（原核、cDNA、无内含子基因）

```bash
./protlocus -S -t 16 genome.fa proteins.fa > aln.paf
```

---

## 10. GPU 使用指南

### 10.1 设备上运行什么

`--gpu` 把三类计算交给 OpenCL 设备，其余（索引、种子、建链、精修、输出）仍在 CPU 多线程上：

| 计算 | 触发条件 | 说明 |
| --- | --- | --- |
| 端延伸 DP（`ns_global_gs16b` 的 EXT_LEFT/RIGHT） | `--gpu`，全部请求 | 2.0.0 新流水线；输入准备也在设备完成；按行分块推进，多个查询的请求共享一次 launch |
| 锚点间 gap DP（`ns_gpu_global`） | `--gpu` 且 nt×aa ≥ `--gpu-min-cells` | 1.0.0 已有；条带式 DP，含 CIGAR |
| HMM profile Viterbi | `--gpu` 且 `MP_HMM_GPU=1`，profile ≤ 1023 match 状态 | 每候选区域一个 work-group；默认关闭，见 10.4 |

设备不能接受的请求（过大、打分参数与批内首个请求不一致、OpenCL 调用失败、内核编译失败）自动回退到 CPU SIMD 内核。端延伸与 HMM 打分的结果与 CPU 内核逐位一致；带 CIGAR 的 gap DP 得分一致，但等分路径的选择可能不同（第 15 节）。

### 10.2 端延伸流水线的结构

1. 每条查询的 `mp_align` 把端延伸请求加入批（`ns_ext_batch_add`），阶段化推进，不阻塞其他候选。
2. flush 时请求进入共享设备 arena；一个常驻执行线程用 `NS_EXT_GPU_QUEUES`（默认 8，最多 16）个命令队列轮转，每次 launch 推进所有在途请求若干行，x-drop 终止的请求即时退出，其 lane 由后续请求补位。
3. NVIDIA 设备上编译带内联 PTX 的 warp 同步内核（32-lane 子组用 `bar.warp.sync` + `shfl.sync`，更宽子组用命名 barrier）；编译失败或 `NS_EXT_GPU_NO_WARP=1` 时使用可移植 work-group 内核。
4. 结果按请求 id 取回，调用方在等待期间继续处理其他候选。

### 10.3 什么时候值得用 GPU

- 查询 ≤ 1000 条：GPU 与 CPU 持平（设备未饱和，流水线开销与收益相抵）。
- 查询数千条以上：GPU 更快；5000 profile 时 1.28×。此时 `device_s ≈ wall`，设备是瓶颈——更快或独占的 GPU 会继续放大收益，多进程共享一块 GPU 则会互相拖慢。
- 不使用 `--accuracy` 时端延伸占比更低，GPU 收益也更小。

### 10.4 HMM 打分上 GPU（`MP_HMM_GPU=1`）

HMM Viterbi 在总时间中占比 <7%，而它使用独立的 OpenCL 上下文、逐查询阻塞 launch，会与端延伸流水线争抢设备；在 56 线程 2000 profile 基准中反而慢 8 s，因此默认关闭。只在端延伸不是瓶颈（如不用 `--accuracy`）或有第二块 GPU（`--gpu-device`）时尝试开启。

### 10.5 环境变量

| 变量 | 作用 |
| --- | --- |
| `NS_EXT_GPU_QUEUES=N` | 端延伸执行器命令队列数（默认 8，上限 16） |
| `NS_EXT_GPU_DEBUG=1` | 退出时输出一行 `[ns_ext_gpu] flushes= launches= wgs= warp_wgs=(on/off) gpu_reqs= cpu_reqs= upload_MB= cells= prep_s= upload_s= device_s= cpu_s= wait_s=`；`=2` 每次 launch 打印 rows/lanes/in_flight；`=3` 另在主机上复算设备侧输入准备并比对 |
| `NS_EXT_GPU_NO_WARP=1` | 不编译 warp 同步内核 |
| `NS_EXT_GPU_STRICT=1` | 本应上 GPU 的端延伸请求回退 CPU 时直接 abort（用于定位回退原因） |
| `MP_HMM_GPU=1` | HMM profile 打分上 GPU |
| `MP_HMM_GPU_DEBUG=1` | 退出时输出 `[mp_hmm_gpu] launches= regions= nt_MB= wait_s=` |
| `NS_GPU_DEBUG_CHECK=1` | 校验每个子比对 CIGAR 的 nt/aa 跨度，不一致时把序列与全部打分参数 dump 到 stderr（任何后端都可用） |
| `OCL_ICD_VENDORS=DIR` | OpenCL ICD loader 的标准变量，指定厂商 ICD 目录 |

### 10.6 读懂 `[ns_ext_gpu]` 统计

- `gpu_reqs` / `cpu_reqs`：设备执行与回退 CPU 的端延伸请求数。`cpu_reqs` 主要来自超长请求（`npad` 超过设备上限），全库扫描中占 0.23%。
- `device_s`：设备忙碌时间；接近 wall time 说明设备是瓶颈。
- `prep_s` + `upload_s`：主机侧准备与上传；正常应远小于 `device_s`。
- `wait_s`：调用方等待结果的累计时间（跨线程求和，可以大于 wall time）。
- `warp_wgs=(on)`：warp 内核已启用。

### 10.7 GPU 候选预筛（`--gpu-prefilter`，仅 CUDA）

粗建链之后、精修之前，把每条候选链的锚点打包成批，一次调用交给 CUDA
无 gap X-drop 内核打分，按分数只保留前 `--gpu-prefilter-topk` 条候选链
（同时保证每个目标序列/链方向分组至少 `--gpu-prefilter-diversity` 条不被
挤掉），减少后续精修/比对阶段要处理的候选数。

- 仅在普通 FASTA 取种/建链路径生效；`--prescreen`、`--cand-table` 等注入
  候选表的路径不经过这一步（候选已经是外部给定的，不需要预筛）。
- 需要 `make cuda=1` 构建；CPU-only 或 `make gpu=1`（OpenCL）构建下
  `--gpu-prefilter` 会打印一次性警告并被忽略，行为等同不加该参数。
- `--gpu-prefilter-adaptive` 用最优/次优候选链的分差在 {1,2,4,8,
  `--gpu-prefilter-topk`} 中选 K，分差大（明显只有一个好候选）时用小 K，
  分差小（多个候选接近）时用大 K。
- `--profile` 模式下 `[M::profile]` 行新增
  `prefilter_candidate/prefilter_survivor/prefilter_k1/k2/k4/k8/prefilter_s`
  字段，用于观察预筛前后的候选数与耗时。
- 这是 3.0.0 之前原型（跨查询批处理、`worker_coarse_for`/`worker_finish_for`
  两阶段流水线）的简化重实现：改为每条查询单独调用一次 GPU 打分，不做
  跨查询打包，因此吞吐特征与早期基准数据不完全可比。

### 10.8 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| `GPU unavailable (clGetPlatformIDs failed with OpenCL status -1001)` | ICD loader 找不到厂商 ICD。常见于链接到 Conda 的 libOpenCL；用系统 gcc 重编，或设 `OCL_ICD_VENDORS=/etc/OpenCL/vendors` |
| `GPU unavailable (OpenCL backend not compiled...)` | 用的是 CPU 版 `protlocus`；换 `protlocus`（以 `make gpu=1` 构建） |
| `[ns_ext_gpu] build failed:` + 编译日志 | 设备不接受内核源码。先试 `NS_EXT_GPU_NO_WARP=1`；仍失败请附日志反馈 |
| 输出与 CPU 版有少量差异 | 预期行为（等分路径），见第 15 节 |
| GPU 版比 CPU 版慢 | 查询太少、GPU 被其他进程占用、或未用 `--accuracy`；用 `NS_EXT_GPU_DEBUG=1` 看 `device_s` |
| `Assertion ... mp_extra_cal` | 2.0.0 已修复此断言的已知触发路径；若再出现，用 `NS_GPU_DEBUG_CHECK=1` 重跑并保存 dump |
| `--gpu-prefilter` 被忽略并有警告 | 当前二进制不是 `make cuda=1` 构建；换用 CUDA 构建，或去掉该参数 |

---

## 11. 性能基准

测试环境：共享计算节点（load 70–125），56 线程，NVIDIA L20（与其他用户共享），小鼠 GRCm39 `.mpi` 索引，Swiss-Prot 聚类代表序列的 HMMER3 profile（平均 `LENG` 395，最长 35213，4952 个 >1023），子集按簇大小降序取前 N 个。命令：`protlocus[-gpu] -t56 --accuracy [--gpu] genome.mpi topN.hmm`（GPU 运行另设 `NS_EXT_GPU_DEBUG=1`）。

### 11.1 HMM 库 GPU vs CPU

| Profile 数 | CPU wall | GPU wall | GPU `device_s` | GPU/CPU | 说明 |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 25.6 s | 25.4 s | 17.8 s | 1.0× | 设备未饱和 |
| 5,000 | 153.8 s | 120.6 s | 103.7 s | 1.28× | 设备成为瓶颈 |
| 83,230（全库） | 2296 s（38.3 min） | 2206 s（36.8 min） | 1881.5 s | 1.04× wall，2.58× CPU time | 见 11.2 |

### 11.2 全库扫描详情（83,230 profile，15 GB）

| 指标 | CPU | GPU |
| --- | --- | --- |
| Wall | 2295.95 s | 2206.17 s |
| CPU time | 105,946 s（平均 46 核忙） | 41,049 s（平均 18.6 核忙） |
| Peak RSS | 30.9 GB | 31.6 GB |
| PAF 行数 | 119,540 | 119,536 |
| 有命中的 profile | 80,850 | 80,849 |
| `[WARNING]` / 断言 | 0 | 0 |

GPU 运行的 `[ns_ext_gpu]` 统计：`flushes=730755 launches=1282023 wgs=12312744 warp_wgs=4933620(on) gpu_reqs=21407608 cpu_reqs=49679 upload_MB=289408.7 cells=3.73e+13 prep_s=56.5 upload_s=152.5 device_s=1881.5 cpu_s=1376.9 wait_s=38351.2`。

解读：

- 2140 万个端延伸请求（3.7×10¹³ 个 DP cell）中 99.77% 在设备上完成；回退 CPU 的 49,679 个（0.23%）是超长 profile（`npad` 超过设备上限）。
- `device_s` = 1881 s ≈ 85% wall：这块与其他用户共享的 L20 是瓶颈，所以 wall 只快 4%；但 CPU 时间从 10.6 万 s 降到 4.1 万 s（-61%）——在负载 70–125 的共享节点上，GPU 版把约 27 个核还给了其他作业。独占 GPU 或更快的设备会直接缩短 wall。
- 首次全库 GPU 运行在第 14 批触发 `mp_extra_cal` 断言，定位到回溯 `D` 长度 bug（第 1 节）并修复后重跑，上表为修复后的结果。

PAF 差异（GPU vs CPU）：

| 类别 | 数量 | 占比 |
| --- | --- | --- |
| 位点相同（前 9 列一致）、区内比对路径不同（`cg`/`cs`，连带 `ms`/`np`/第 10–11 列；`AS` 不同 23 行） | 768 行 | 0.64% |
| 位点边界不同 | 46 行 / 22 个 profile | 0.04% / 0.03% |
| 仅 CPU 命中的 profile | 1（sp_Q8ILR9_YPF17_PLAF7） | 0.001% |
| 完全一致 | 118,700+ 行 | 99.3% |

这些差异来自带 CIGAR 的 gap DP 在等分/近等分路径上的不同选择（端延伸 DP 只返回得分与端点，GPU/CPU 逐位一致），在 warp 内核之前的 1.0.0 GPU 路径中已存在；GPU 重跑与自身逐字节一致。

### 11.3 `--accuracy` 与 AVX2（1.0.0 起）

四基因组、1114 蛋白、56 线程、三次取中位：`--accuracy` 默认 65.2 s，穷举 158.3 s，召回相同；AVX2 自动派发使端延伸 DP 时间减少 12.4–16.6%，输出逐字节一致。

### 11.4 单元性能

warp 内核使单 warp 端延伸每行耗时 950 → 540 ns；top200 -t16：10.2 → 7.8 s；top2000 -t56：57 → 50 s（同负载 CPU 67 s）。发布包中的二进制在 top200（`-I -u -t16 --accuracy`）上：`protlocus-gpu --gpu` 7.4 s / 3.6 GB，`protlocus` 11.2 s / 3.3 GB，247 行 PAF 中 2 行为等分路径差异。

---

## 12. `--profile` 统计字段

运行结束时 stderr 输出一行 `[M::profile] key=value ...`：

| 组 | 字段 | 含义 |
| --- | --- | --- |
| 输入 | `query`, `aa` | 查询数、氨基酸总数 |
| 种子 | `seed`, `anchor`, `anchor_rescued` | 种子数、锚点数、`--accuracy` 救回的锚点数 |
| 建链/精修 | `chain`, `refine_candidate`, `refined`, `exact_anchor`, `refine_kmers`, `refine_pairs`, `refine_pairs_dropped`, `refine_buckets_capped` | 链数与第二轮精修工作量 |
| DP 候选 | `dp_candidate`, `dp_rescued`, `dp_ext_eligible`, `dp_ext_retry`, `dp_ext_fallback`, `dp_realign`, `dp_realign_kept` | 进入 DP 的候选、无锚点救回、端延伸重试/回退、可疑剪接点重比对与被采纳数 |
| DP 调用/规模 | `align_gap_calls`, `align_cells`, `align_ext_calls`, `align_ext_cells`, `align_ext_retry`, `align_full_calls`, `align_full_cells`, `align_ext_calls_unguarded`, `align_ext_cells_unguarded` | gap DP / 端延伸 DP / 全窗口 DP 的调用次数与 cell 数 |
| GPU | `gpu_dp`, `gpu_launch`, `gpu_h2d_s`, `gpu_kernel_s`, `gpu_d2h_s` | gap DP 上 GPU 的次数、launch 数与三段耗时（端延伸流水线的统计见 `NS_EXT_GPU_DEBUG`） |
| 输出 | `output` | 输出比对数 |
| 耗时 (s) | `sketch_s`, `anchor_s`, `prechain_s`, `chain_s`, `refine_s`, `refine_sketch_s`, `refine_pair_s`, `refine_chain_s`, `align_s`, `align_ext_s`, `align_gap_s`, `align_full_s`, `output_s`, `wall_s` | 各阶段跨线程累计时间；`wall_s` 为墙钟 |
| 内存 | `km_max_blocks`, `km_max_capacity` | kalloc 空闲链最长长度与 arena 最大容量 |

---

## 13. Python 辅助工具

`tools/`，Python 3 标准库即可，`python3 tools/<name>.py --help` 查看参数。

| 脚本 | 用途 |
| --- | --- |
| `filter_paf.py` | 过滤 PAF：`--min-cov`、`--require-complete`、`--min-ra`、`--max-fs`、`--max-st` |
| `paf_metrics.py` | 子命令 `summary` / `compare` / `regressions`：PAF 质量与召回指标、基线-候选对比、回归列表 |
| `evaluate_accuracy.py` | 与真值注释（`--truth`）比较预测（`--prediction`），支持边界容差与位点重叠阈值，可输出 JSON |
| `junctions_to_spsc.py` | 把 STAR `SJ.out.tab`（`--format star`）或 BED6（`--format bed6`）剪接证据转换为 `--spsc` 格式，可设读数阈值与权重 |
| `run_accuracy_benchmark.py` | 批量运行多模式/消融（`--ablations`）基准并汇总 |
| `run_large_query_batch.py` | 大查询集去重、按长度排序、分批运行并恢复原顺序 |
| `mmseqs2cand.py` | 把 mmseqs2 profile-vs-ORF 搜索结果转换为 `--cand-table` 候选表（含 E 值列） |
| `protlocus_genemodel.py`（当前源码功能） | 由 HMM 库命中（`final.paf`）构建基因模型，输出 GFF3 + 蛋白 FASTA（见 9.4c） |
| `evaluate_genemodel.py`（当前源码功能） | 与 RefSeq/TAIR CDS 在 nt/外显子/基因层级比较预测的 GFF3，可输出 JSON |
| `hmmrecheck.py`（5.0.0） | 候选复核的 hmmsearch 交叉验证版：解析 `--trans` 输出的 PAF+`##STA` 肽段，按家族 hmmfetch+hmmsearch 重打分，按 domain i-Evalue 过滤（需 HMMER3；原生版见 `recheck.c` → `protlocus-recheck`，9.4b 节） |
| `protlocus_hmm_pipeline.sh`（5.0.0） | 一键跑「整库扫描 + HMM 复核」两段流水线，当前源码中 `-G genome.fa` 追加第 3 段基因模型（bash；默认原生 `protlocus-recheck`，`-R hmmer` 切 hmmsearch 交叉验证；protlocus 参数取基准冻结值） |

---

## 14. 常见问题

**Q: 结果和上游 miniprot 不完全一样？**
不加 `--accuracy`、`--gpu`、`.hmm` 时应当一致。2.0.0 修复的回溯 `D` 长度 bug 只在以 deletion 开头的子比对上改变输出，而 CPU 路径在实践中不会产生这种子比对；DPP3 样例输出与 1.0.0 及上游一致。

**Q: `.hmm` 文件被当成 FASTA 解析报错？**
文件名必须以小写 `.hmm` 结尾；`.HMM`、`.hmm.gz` 不识别。gzip 压缩的 HMM 库请先解压。

**Q: 一个 profile 输出多行？**
与蛋白查询相同：每个位点一行，`Rank`/顺序按 `AS` 排序。`--outn 1` 只保留最优位点；`--outs 1.0` 只保留与最优同分的。

**Q: 需要多大内存？**
索引载入 + 每线程 mini-batch 工作区。参考：小鼠基因组、56 线程、全库 HMM 扫描峰值 31–33 GB；16 线程约减半。`-K` 减小 mini-batch 可降低内存。

**Q: `protlocus`（以 `make gpu=1` 构建） 不加 `--gpu` 会用 GPU 吗？**
不会，行为与 `protlocus` 一致。

**Q: 可以多进程共享一块 GPU 吗？**
可以运行，但设备是瓶颈时会互相拖慢；优先用一个进程 + 更多线程。

**Q: 如何只重新比对某几个 profile？**
用 `awk` 按 `//` 切分或用 HMMER 的 `hmmfetch` 提取后再拼接。

---

## 15. 已知限制

- DP 初始条件在理论上不完全正确，极少数情况下残基比对可能非最优（继承自上游）。
- 非剪接比对（`-S`）的支持仍有改进空间。
- `--gpu` 下约 0.7% 的 PAF 行与 CPU 版不同：绝大多数是同一位点内的不同等分比对路径（`cg`/`cs`/`ms`/`np`/第 10–11 列），约 0.04% 的行边界不同，全库 8.3 万 profile 中有 1 个仅 CPU 命中（第 11.2 节）。GPU 重跑逐字节一致。
- HMM 查询依赖一致序列取种：一致序列在基因组中没有可识别 k-mer 的家族可能漏检。
- HMM GPU 打分限 ≤1023 个 match 状态；更长 profile 自动走 CPU。
- CUDA 构建覆盖 gap DP 与 `--gpu-prefilter` 无 gap 预筛内核，无端延伸流水线与 HMM GPU 打分；`--gpu`/`--gpu-device`/`--gpu-min-cells` 等 OpenCL 专属参数在 CUDA 构建下无效。
- `--gpu-prefilter` 是 3.0.0 之前原型的简化重实现（每查询单独调用，不做跨查询批处理），尚未做过与旧基准数据的吞吐/精度对比（第 4 节新增）。
- `--accuracy` 的默认阈值来自四个脊椎动物基因组的调参，不是普适常数；用于其他分类群前请在有真值的数据上验证。

---

## 16. 从 1.0.0 / 2.0.0 升级

**历史 5.0.0 → 当前源码**：命令行完全兼容；FASTA/单 profile 路径输出不变。仅一处默认行为变化：`--prescreen` 且基因组 ≤ 50 Mb 时自动启用小基因组预设（输出变多、召回更高，需配合 `protlocus-recheck`）；要复现 5.0.0 输出请加 `--small-genome=off`。新增流水线 `-G/-X` 与 `protlocus_genemodel.py`、`evaluate_genemodel.py`。

**从 2.0.0 升级：**

- 命令行完全向后兼容；`.mpi` 索引可直接使用；FASTA 与单 profile 查询输出不变。
- HMM 命中新增默认 `--acc-min-ps=60` 下限：若需完全复现 2.0.0 的 HMM 输出，加 `--acc-min-ps=0`。
- 整库扫描推荐改用 `--prescreen --tier=...`；基于 `--thr-table` 的按家族 decoy 阈值协议仍可用但不再推荐。
- 若基于源码二次开发：`mp_mapopt_t` 新增 `ps`（`mp_psopt_t`）、`prescreen`、`cand_max_evalue`、`acc_min_ps` 等字段；新增 `prescreen.c`（`mp_prescreen_build/batch/destroy`）。改动选项结构体后请 `make clean` 重编。

**从 1.0.0 升级：**

- 命令行完全向后兼容；1.0.0 的索引 `.mpi` 可直接使用。
- 新增二进制 `protlocus`（以 `make gpu=1` 构建）；1.0.0 的建议"GPU 慢于 CPU"已过时。
- `MP_HMM_NO_GPU` 已移除，HMM GPU 打分改为 `MP_HMM_GPU=1` 显式开启。
- 若基于 1.0.0 源码二次开发：`mp_align()` 末尾新增 `mp_hmm_batch_t *hb` 参数（传 NULL 即内联打分）；`nasw.h` 新增 `ns_ext_batch_*` 接口。

---

## 17. 引用与许可

ProtLocus 基于 miniprot；使用时请引用原始论文：

> Li H. Protein-to-genome alignment with miniprot. *Bioinformatics*. 2023;39:btad014. doi:10.1093/bioinformatics/btad014

代码以 MIT 许可证发布（`LICENSE.txt`）；上游版权声明（Copyright (c) 2022- Dana-Farber Cancer Institute）按许可证要求保留。ProtLocus 与 miniprot 原作者无关联。
