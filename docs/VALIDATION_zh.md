# 本地构建与验证记录

验证对象为准备公开发布的 ProtLocus 0.1.0，保留全部 CPU/OpenCL/CUDA 源码。

## 已实际执行

在本机 Docker 的 Linux amd64 环境中执行：

- CPU：Ubuntu 22.04 / GCC 11，编译、C 测试、21 项 Python 测试、版本和 DPP3 样例哈希检查通过。
- OpenCL：Ubuntu 22.04 / GCC 11 / OpenCL loader，编译、C 测试的 CPU 部分、21 项 Python 测试、版本和默认 CPU 路径样例检查通过。原生 GPU 测试因没有设备而明确跳过。
- CUDA：CUDA 12.4.1 / Ubuntu 22.04，生成 sm_70、75、80、86、89、90 和 compute_80 PTX 成品；C 测试的 CPU 部分、21 项 Python 测试、版本和默认 CPU 路径样例检查通过。原生 GPU 测试因没有可用驱动/设备而明确跳过。
- 发布材料测试覆盖：源码与配方 SHA256 对应、三个包的配方生成、版本/标签不一致拒绝、未提交源码拒绝、完整包三种程序与全量源码、归档解包路径检查及文件校验。
- 三套 Conda 配方与五个 GitHub 工作流的 YAML 结构已检查。
- 已在 Linux amd64 的真实 conda-build 环境中构建出三个 `.conda` 包，并通过各自的 Conda 安装测试。
- 已创建一个新的 Conda 用户环境，同时安装 `protlocus`、`protlocus-gpu`、`protlocus-cuda`，检查命令共存、版本、库链接、DPP3 输出哈希以及配套工具入口，全部通过。
- 安装后的 OpenCL/CUDA 程序分别从 Conda 环境加载 `libOpenCL.so.1` 和 `libcudart.so.12`。无设备时要求真实 GPU 的验证均按预期失败。
- 修正 Conda 的 CPPFLAGS 覆盖 GPU 开关问题；修正 prefixed readelf 检查和 CUDA 编译器宏配置，并将 CUDA 12.4 的 Conda GCC 工具链调整为兼容的 GCC 12。
- 最终源码的测试数量增加到 22 项 Python 测试（新增 GPU 编译开关保留检查）；最终本机 CPU 编译、C 测试及这些 Python 测试通过。
- 最终完整包解压后，CPU/OpenCL/CUDA 三种程序均通过全部文件校验、版本及小样例检查。无真实 GPU 时，两个 GPU 后端的 `--require-gpu` 检查均按预期失败，没有把 CPU 回退算作 GPU 验证成功。
- 已检查 CUDA 成品实际包含六种架构的 cubin 与 compute_80 PTX，OpenCL/CUDA 成品分别链接对应运行库。

## 仍需在对应环境验证

- 真实 OpenCL/CUDA 设备上的计算结果和性能：使用 `verify.py --require-gpu` 或手动 GPU runner 工作流。
- Bioconda 官方环境的 lint、构建及维护者审核尚未执行；本地 conda-build 构建、依赖求解和独立安装已完成。
- 远程 GitHub Actions 及 Bioconda 官方构建/审核：本次只准备本地项目，没有发布远程内容。

原始文档内的 NVIDIA L20 等性能数据仍为历史测试记录，不是本次重新测量。

## glibc 2.28 兼容重建

三套 Conda 构建 1 使用 GCC 12 与 sysroot 2.28，已在 Rocky Linux 8（实际 glibc 2.28）安装共存，并通过版本、动态库链接和 DPP3 默认 CPU 路径哈希验证。三个后端主程序均未引用高于 GLIBC_2.28 的符号。实际 GPU 运算尚需北大 HPC 的显卡节点验证。独立原生 tar.gz 的 glibc 2.35 基线没有改变。
