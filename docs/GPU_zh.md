# GPU 支持与构建

GPU 功能保留在源码中。源码项目不附带原始 Linux 预编译程序 `protlocus-gpu`；该文件仍在原始发布包中，但其版本和运行环境属于旧发布包，不应冒充当前版本。

## GPU 后端的范围

- **OpenCL（`make gpu=1`）**：包括端延伸流水线、gap DP，以及可选 HMM profile GPU 打分。需要 OpenCL loader 和厂商驱动/ICD。
- **CUDA（`make cuda=1`）**：包含 gap DP 及 `--gpu-prefilter` 无 gap 候选预筛；需要 nvcc 和 CUDA toolkit。不是与 OpenCL 完全相同的执行路径。
- **CPU**：始终保留，不需要 GPU 环境。

相关文件包括 `nasw-opencl.c`、`nasw-ext-gpu.c`、`nasw-ext-kernel.h`、`hmm-gpu.c`、`nasw-cuda.cu` 和 `test/test_gpu.c`。这些文件未被本次整理删除。

## 同时保留 CPU 和 OpenCL 程序

在 Linux 及配置好的 OpenCL 环境中，从项目根目录运行：

```bash
# 保存 CPU 程序到独立目录，避免后续 make clean 删除它
make clean
make -j2
mkdir -p build-cpu
cp protlocus protlocus-recheck build-cpu/

# 构建 OpenCL 程序；默认输出文件名仍是 protlocus
make clean
make -j2 gpu=1
mkdir -p build-opencl
cp protlocus build-opencl/protlocus-gpu

# 有兼容设备时运行 GPU 测试
make gpu=1 check

# 使用 OpenCL 程序；不加 --gpu 时使用 CPU 路径
./build-opencl/protlocus-gpu --gpu genome.fa proteins.fa > result.paf
```

OpenCL HMM profile GPU 打分默认关闭，可显式启用：

```bash
MP_HMM_GPU=1 ./build-opencl/protlocus-gpu \
  -I -u -t 8 --accuracy --gpu genome.mpi library.hmm > result.paf
```

这条路径需要自行验证速度和结果；GPU 对小输入不一定有优势。完整参数及诊断环境变量见 [完整手册](MANUAL_zh.md) 第 10 节。

## CUDA 与候选预筛

```bash
make clean
make -j2 cuda=1 CUDA_ARCH=sm_80
./protlocus --help
```

将 `CUDA_ARCH` 改成目标设备对应的架构。需要时显式传 `NVCC=/path/to/nvcc` 和 `CUDA_HOME=/path/to/cuda`。CUDA 候选预筛由 `--gpu-prefilter` 启用；具体限制和参数见完整手册第 10.7 节。OpenCL 的 `--gpu-device` 等选项不能视为 CUDA 的通用选项。

## 当前验证与打包状态

本地已实际编译 CPU、OpenCL、CUDA 三种后端，并通过默认 CPU 路径和样例测试；容器没有真实 GPU，因此设备测试明确跳过。详细状态见 [验证记录](VALIDATION_zh.md)。原始文档中的 L20 等结果为历史测试记录。

Conda 提供三套配方：`protlocus` 为 CPU 主包，`protlocus-gpu` 为 OpenCL 扩展包，`protlocus-cuda` 为 CUDA 扩展包。扩展包依赖同版本 CPU 主包，分别安装 `protlocus-gpu` 和 `protlocus-cuda`，避免覆盖 `protlocus`。

OpenCL 配方声明 `ocl-icd` 依赖；CUDA 配方使用 CUDA 编译器并声明 CUDA runtime 依赖。GPU 驱动及厂商 ICD 不随包分发，必须由运行环境提供。CUDA 配方通过配置选择 12.4 编译器和 GCC 12，覆盖 sm_70 到 sm_90 并保留 compute_80 PTX。

Release 提供三种二进制包及全量总包，总包包含全部源码和三种主程序。普通 GitHub runner 没有 GPU，构建成功只证明编译、链接及默认 CPU 路径通过。`.github/workflows/gpu-hardware.yml` 可在带有 `self-hosted, linux, x64, gpu` 标签的自托管 runner 上手动执行，它将缺少设备视为失败，而不是跳过。

解压二进制包后，执行 `python3 verify.py --backend opencl --require-gpu` 或 `--backend cuda --require-gpu`，验证实际设备初始化、样例和原生 GPU 测试。未加 `--require-gpu` 仅检查文件与 CPU 默认路径。
