# ProtLocus 完整功能与使用说明


本项目提供蛋白质/蛋白家族到基因组的定位、原生 HMM 复核以及辅助基因模型构建，基于 miniprot 0.18-r281。当前独立公开版本准备为 0.1.0。

这份说明保留原始快照中的详细功能、命令、历史开发说明和基准数据。历史编号仅用于记录开发过程，不是当前公开版本号；基准数据未在本次整理中重新测量。源码、工具、测试与 GPU 后端都保留。

完整手册见 [MANUAL_zh.md](MANUAL_zh.md)，GPU 专题见 [GPU_zh.md](GPU_zh.md)，历史技术说明见 [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md)。以下命令在项目根目录执行。

## 目录内容

- 根目录：C/CUDA 源码、头文件、Makefile、许可证、英文 man page。
- `tools/`：HMM 复核流水线、基因模型构建和评估等 Python 工具。
- `test/`：C/Python 测试和小样例数据。
- `docs/`：完整中文手册、技术报告、历史开发记录和 GPU 使用说明。
- `packaging/`：CPU、OpenCL、CUDA 三套 Conda 配方模板与安装测试。

源码项目不附带旧的预编译程序、`.o`、`.a`、旧构建清单与旧校验脚本。CPU、OpenCL 和 CUDA 的源码及构建开关均保留。

## 构建与检查

```bash
make -j2
make check
./protlocus --version
./protlocus-recheck -v
```

OpenCL：`make clean && make -j2 gpu=1`；CUDA：`make clean && make -j2 cuda=1`。各后端默认都生成名为 `protlocus` 的程序，切换前需清理；如要同时保留 CPU/GPU 程序，可分别复制并重命名，见 [GPU 说明](GPU_zh.md)。

## 最常用的命令

```bash
./protlocus -t 16 -d genome.mpi genome.fa                                   # 建索引
./protlocus -I -t 16 genome.mpi proteins.fa > aln.paf                       # 普通模式（同上游 miniprot）
./protlocus -I -u -t 16 --accuracy --gff genome.mpi proteins.fa > pred.gff  # 高召回 + GFF3
./protlocus -I -u -t 16 --accuracy genome.mpi family.hmm > family.paf       # HMMER3 profile 查询（单个/少量）
cat lib/*.hmm > library.hmm
./protlocus -I -u -t 56 --accuracy --prescreen --tier=balanced genome.mpi library.hmm > library.paf   # 整库扫描（推荐；紧凑基因组如酵母用 --compact 代替 --accuracy）
./protlocus -I -u -t 56 --accuracy --gpu genome.mpi library.hmm > library.paf                    # 整库全基因组取种 + GPU
bash tools/protlocus_hmm_pipeline.sh -g genome.mpi -m library.hmm -o out -t 56                   # 5.0.0：整库扫描 + HMM 复核（最终命中带统计显著性）
bash tools/protlocus_hmm_pipeline.sh -g genome.mpi -m library.hmm -o out -t 56 -G genome.fa      # 当前源码：再加第 3 段，输出 out.genes.gff3 / out.genes.faa（原核加 -X "--no-introns"）
```

## 当前功能：基因模型与小基因组预设

- **基因模型构建（第 3 段）**：`tools/protlocus_genemodel.py`（纯标准库 Python ≥ 3.6，`-t` 按 contig 多进程）把第 2 段 `final.paf` 转成 GFF3（`gene/mRNA/exon/CDS`）+ 蛋白 FASTA。流程：命中 → 带密码子相位的外显子块（同链重叠命中按框/结构合并或取显著性高者）→ 相邻证据单元连接（同框直接融合，否则在 GT/GC..AG 内含子上做保持读框的 DP，须优于「两个独立基因」基线 `--link-min`）→ 两端按 ORF 延伸（3' 到首个同框终止子，5' 到累计编码得分最佳的 ATG）。剪接位点 PWM、内含子长度分布、密码子/二联密码子对数比均由基因组自身命中自训练；原核用 `--no-introns`。
  - 一键：`protlocus_hmm_pipeline.sh ... -G genome.fa [-X "额外选项"]` 产出 `OUTPREFIX.genes.gff3/.genes.faa/.stage3.log`；`-G` 需基因组 FASTA（不是 `.mpi`）。
  - 评估：`tools/evaluate_genemodel.py --ref ref.gff3 --pred pred.gff3 [--paf final.paf]`，在核苷酸/外显子/基因三个层级与 RefSeq/TAIR CDS 比较。
  - 实测（Pfam-A 命中，E≤1e-5，对 RefSeq CDS，默认参数）：酵母 nt 灵敏度 0.84 / 特异度 0.99，87.7% 预测基因 CDS 结构与参考一致；大肠杆菌（`--no-introns`）0.89 / 0.98；拟南芥（内含子丰富）0.55 / 0.81 且仅约 22% 预测基因完全一致——内含子丰富基因组请把输出当作「以蛋白域为锚的基因骨架 / 提示」（如喂给 Augustus），不是最终模型；没有任何 Pfam 命中的基因找不到。
- **小基因组召回预设**：`--small-genome=auto|on|off`（默认 auto）与 `--small-genome-size=NUM`（默认 50M）。`--prescreen` 且基因组总长 ≤ 阈值（或 `on`）时自动改用 `--ps-kmer=5 --acc-min-ps=0 --acc-min-score=0 --acc-min-pos-ratio=0`，命令行显式给出的这四项保持不变；弱命中交给 `protlocus-recheck`（E≤1e-5）裁决。大肠杆菌/酵母结构域召回 0.7555→0.8023 / 0.7367→0.7809，纯度不变、分钟级。`--small-genome=off` 与 5.0.0 输出逐字节一致。不要与放宽的 sensitive 档预滤同时使用（候选爆炸会抬高复核 Z、降低最终召回）。

## 5.0.0 要点

- **候选 + HMM 复核流水线（原生实现，零 HMMER 依赖）**：ProtLocus 定位为极速高灵敏候选生成器，复核段给每条候选以统计校准的显著性。
  - 第 1 段：`protlocus --prescreen --tier=... --trans` 整库扫描，每条命中后输出一行 `##STA`——沿比对路径（跨内含子、跳过移码）翻译的基因组肽段。
  - 第 2 段：**`protlocus-recheck`（`recheck.c`，原生 C 复核器）**把每条候选肽段**只对它自己家族的 profile** 跑完整 Forward 算法（plan-7 统计模型的独立实现：multihit local 配置、占用率加权入口、目标长度 N/C/J 模型、log 空间 Forward、null1 几何长度模型），用 profile 自带的 `STATS LOCAL FORWARD tau lambda` 校准行换算 E 值并按 E≤1e-5 过滤。全程不调用任何 HMMER 程序。
  - 为什么又快又准：六框全库 hmmsearch 是「全部 profile × 全部 ORF」交叉积，这里每条肽段只打一个模型（Pfam-A 下约为交叉积的两万分之一），且候选肽段总量远小于六框 ORF；`##STA` 肽段是剪接后的，跨内含子的域不再像单帧 ORF 那样被截断。
  - 输出：通过复核的 PAF（追加 `fB:f:` Forward bits、`fE:f:` E 值），`-a` 保留落选行并标 `hp:i:0`；`-T` 输出逐候选 TSV（肽长、读框内终止子数=假基因信号、Forward 统计）。
  - E 值刻度：`-Z` 默认取唯一候选肽段数；要与六框口径严格可比，传该物种六框 ORF 数。
  - 一键封装 `tools/protlocus_hmm_pipeline.sh`（protlocus 参数默认取基准冻结值 `-I -u --outs=0 -N 1000 --outn=1000 --acc-pri-ratio=0.1`），各段带 `/usr/bin/time -v` 资源日志;`-R hmmer` 可切换到 `tools/hmmrecheck.py`（逐家族调 hmmfetch/hmmsearch，按 best domain i-Evalue 过滤），用于与 hmmsearch 交叉验证原生引擎。
  - 构建：`make protlocus-recheck`（仅需 gcc + zlib；源码项目需要自行构建，不附带旧发布包的预编译复核器）。

## 4.0.0 要点

- **`--gpu-prefilter`**（需 `make cuda=1`）：粗建链后用批处理 GPU 无 gap X-drop 内核给候选链打分，只保留前 `--gpu-prefilter-topk`（默认 32）条；`--gpu-prefilter-diversity` 设每个目标/链方向分组的最少保留数，`--gpu-prefilter-xdrop` 设 X-drop 阈值，`--gpu-prefilter-adaptive` 按分差自适应选 K。这是 3.0.0 之前原型（K1/K2/K4/K32/自适应 K 预筛）的简化重实现：改为每查询单独调用，不做跨查询批处理；CPU-only/OpenCL 构建下该参数会被忽略并警告。详见 `MANUAL_zh.md` 10.7 节与 `docs/NEWS.md`。

## 3.0.0 要点

- **`--prescreen`**：整库 HMM 扫描的内置候选位点生成器（六框 ORF 间隔 k-mer 索引 → MMseqs2 风格 profile k-mer 预筛 → 无 gap 对角线 → 带状 gap 比对 → Karlin-Altschul E 值），取代外部 mmseqs + `--cand-table`。酵母 × Pfam-A 38（56 线程，`--compact`）：`--tier=fast` 36 s P0.988 R0.873；`balanced` 1.4 min P0.956 R0.923；`sensitive` 10.5 min P0.936 R0.965。
- **`--cand-table=FILE`** / **`--tier=fast|balanced|sensitive`** / **`--cand-max-evalue`**：外部候选表模式与 E 值分层。
- **`--acc-min-ps`（默认 60）** 与 `--acc-ps-rescue`：HMM 命中的 profile 分数下限；`--thr-table` 等按家族 decoy 阈值仍可用但不再是推荐协议。
- FASTA 与单 profile 行为与 2.0.0 完全一致；DPP3 样例输出哈希不变。

## 2.0.0 要点

- HMMER3 profile 查询（单个或整库 `.hmm`），位置特异 Viterbi 打分。
- GPU 端延伸流水线：大规模库扫描时 GPU 快于 56 线程 CPU（5000 profile 1.28×）；小输入持平。
- 修复回溯末端 `D` 长度 bug（上游 miniprot 同样存在的潜在问题）与 1.0.0 的 HMM GPU 内核静默回退问题。
- 默认行为（不加 `--accuracy`/`--gpu`/`.hmm`）与 1.0.0 及上游输出一致。

## 从源码编译

```bash
make clean && make -j            # CPU
make -j static=1                           # 全静态 CPU
make -j gpu=1 && make gpu=1 check          # OpenCL
make -j cuda=1                             # CUDA（gap DP + --gpu-prefilter）
```

## 引用

Li H. Protein-to-genome alignment with miniprot. Bioinformatics. 2023;39:btad014. DOI: 10.1093/bioinformatics/btad014
