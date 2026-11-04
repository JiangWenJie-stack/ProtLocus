# 安装本地 Conda 频道中的 ProtLocus

此目录是已构建的 Linux x86_64 Conda 频道，包含三个 `.conda` 安装包及频道索引。它尚未上传 Bioconda 或个人公开频道。

构建 2 的三个 Conda 包使用 GCC 12 和 glibc 2.28 sysroot；支持 Linux x86_64、glibc 2.28 或更新版本。独立 tar.gz 二进制仍要求 glibc 2.35+，与此 Conda 频道不同。

在 Linux 上解压整个频道目录，进入包含 `linux-64/`、`noarch/` 的位置，然后执行：

```bash
CHANNEL="file://$(pwd)"
# 同时安装 CPU、OpenCL 和 CUDA 三种程序
conda create -n protlocus --override-channels --strict-channel-priority \
  -c "$CHANNEL" -c conda-forge \
  protlocus=0.1.0 protlocus-gpu=0.1.0 protlocus-cuda=0.1.0
conda activate protlocus
protlocus --version
protlocus-gpu --version
protlocus-cuda --version
```

只需要 CPU 时只指定 `protlocus=0.1.0`。只需要 OpenCL 时指定 `protlocus-gpu=0.1.0`，只需要 CUDA 时指定 `protlocus-cuda=0.1.0`；GPU 包会自动带入同版本 CPU 包。依赖库从 conda-forge 下载，因此首次安装通常需要联网。

激活后可直接运行：

```bash
protlocus genome.fa proteins.fa > cpu.paf
protlocus-gpu --gpu genome.fa proteins.fa > opencl.paf
protlocus-cuda --gpu genome.fa proteins.fa > cuda.paf
```

GPU 驱动和可用显卡由运行机器提供。没有设备时程序会警告并回退到 CPU，不能以“命令执行成功”判断 GPU 已经工作。日志应出现 `[M::gpu] using device`，且不应出现 `GPU unavailable`。

Conda 环境内的 OpenCL loader 需要发现厂商的 ICD 文件。Linux 通常位于 `/etc/OpenCL/vendors`；若未被环境里的加载器发现，可按实际位置设置：

```bash
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
```

这不是安装驱动的替代操作；目录和厂商驱动必须真实存在。CUDA 同样需要兼容的 NVIDIA 驱动。

真实显卡测试仍应在对应设备上执行。完整项目中提供 `tools/verify_conda_install.py` 和独立的 GPU 硬件测试工作流。

```bash
protlocus-example --list
protlocus-example mouse-mt
protlocus-example ecoli
```
