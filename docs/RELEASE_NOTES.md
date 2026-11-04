# ProtLocus 0.1.0

首次以独立项目形式准备公开发布，基于 miniprot 0.18-r281，保留完整源码、工具、测试和技术资料。

- 蛋白质/HMM profile 到基因组定位、PAF/GFF3 输出、原生 HMM 复核与辅助基因模型构建。
- CPU、OpenCL GPU、CUDA 三种后端，提供分别命名的程序和二进制包。
- 全量包包含全部源码、三后端程序、复核器、辅助工具、手册、样例和校验脚本。
- 提供 CPU/OpenCL/CUDA 三套 Conda 投稿配方及实际构建的 `.conda` 包与频道索引；三包在全新环境中共同安装验证通过。
- 普通构建测试与真实 GPU 设备测试分开记录；GPU 驱动不随软件分发。

自动发布附件：源码包，Linux x86_64 的 cpu/opencl/cuda/full 四个包，三套 Bioconda 配方包、可安装的 Conda 频道包及 SHA256SUMS。二进制面向 glibc 2.35+；OpenCL/CUDA 另需对应运行库和驱动。

附带配方不代表已经被 Bioconda 收录。保留 MIT 许可证与 miniprot 上游版权说明。内含子丰富基因组的预测模型宜作为辅助注释证据。

Conda 构建 1：CPU、OpenCL、CUDA 统一使用 glibc 2.28 sysroot 和 GCC 12，配方声明 C 标准库依赖；替代构建 0 的 glibc 2.34 基线。独立 tar.gz 二进制仍要求 glibc 2.35+。GPU 实际设备验证需在有驱动和 GPU 的计算节点进行。

完整离线示例：随包提供小鼠完整线粒体染色体 NC_005089.1、大肠杆菌 K-12 MG1655 完整基因组 NC_000913.3，以及 COX1/lacZ 查询蛋白；新增 `protlocus-example` 命令。FASTA 数据约 4.72 MB，运行无需联网。Conda 构建编号更新为 2。
