# 本地准备完成后的发布步骤

## 你需要操作的部分

1. 将本目录的内容推送到你的 GitHub 仓库根目录，保留 `.github/` 等隐藏文件。网页逐个上传文件容易遗漏隐藏目录，建议使用 Git 或 GitHub Desktop。
2. 查看 Actions 的 **Linux build and tests (CPU, OpenCL, CUDA)**，确认这次提交通过。
3. 打开 **Releases → Draft a new release**，建立新标签 **`v0.1.0`**，选择包含本项目修改的提交。标题填 `ProtLocus 0.1.0`，正文可复制 [发布说明](RELEASE_NOTES.md)。不要勾选 prerelease。
4. 点击 **Publish release**。发布页先出现，自动流程随后运行；等待 **Publish complete source, CPU, OpenCL, CUDA and Conda recipes** 变绿，刷新 Release 页面即可看到附件。

自动流程不需要你提供密码或个人访问令牌，使用 GitHub 自带的短期 `GITHUB_TOKEN`。如果仓库或组织策略禁止 Actions 写入 Release，需要由仓库管理员允许此工作流的 `contents: write` 权限。若组织限制第三方 Actions，也需允许 `actions/checkout`。

## 自动生成的内容

- `protlocus-0.1.0.tar.gz`：从 Release 标签对应提交中导出已跟踪的源码，不包含本地未跟踪文件或编译产物。
- `protlocus-0.1.0-linux-x86_64-cpu.tar.gz`：CPU 程序、复核器、工具和资料。
- `protlocus-0.1.0-linux-x86_64-opencl.tar.gz`：OpenCL GPU 程序、原生 GPU 测试和配套资料。
- `protlocus-0.1.0-linux-x86_64-cuda.tar.gz`：CUDA 程序、原生 GPU 测试和配套资料。
- `protlocus-0.1.0-linux-x86_64-full.tar.gz`：三种程序、全部源码、工具、测试、文档及验证脚本。
- `bioconda-recipe-0.1.0.tar.gz`：包含 CPU/OpenCL/CUDA 三套 `recipes/` 配方。
- `protlocus-0.1.0-conda-channel.tar.gz`：实际构建出的三个 `.conda` 包和频道索引。完整总包也包含 `conda-channel/`。
- `SHA256SUMS`：所有发布压缩包的校验值。

配方的源码地址指向这次 Release 附加的源码包，校验值由实际文件计算；不会跟随 `main` 的后续修改。仓库名称不必固定为 ProtLocus，流程读取真实的 GitHub 仓库地址。

配方维护者默认是点击发布者的个人 GitHub 用户名。若不合适，在 **Settings → Secrets and variables → Actions → Variables** 中添加 `BIOCONDA_MAINTAINER`，值为真正维护配方者的个人 GitHub 用户名。该项不是密码，也不是作者或版权声明。README 中的作者、联系方式可由你按实际情况补充；许可证保留上游声明。

## 用户无需等待 Bioconda 的安装方式

Release 的 Conda 频道附件内含实际安装包，不只是配方。用户在 Linux 解压后，进入含 `linux-64/` 和 `noarch/` 的目录，按附带 `README.md` 用本地频道安装。依赖从 conda-forge 获取；不会自动安装系统显卡驱动。完整命令见 [Conda 安装说明](../packaging/INSTALL_CONDA_zh.md)。

## 后续提交到 Bioconda

下载配方压缩包，解压后把 `recipes/protlocus/`、`recipes/protlocus-gpu/`、`recipes/protlocus-cuda/` 放进你 Fork 的 `bioconda/bioconda-recipes` 仓库，并向官方发起 PR。这里提交配方，不需要再上传整个软件源码。

三套配方都面向 Linux x86_64。CPU 包提供主程序、复核器及辅助工具；OpenCL/CUDA 包依赖同版本 CPU 包，安装不同名称的 GPU 主程序。OpenCL loader、CUDA runtime 等依赖在配方中声明；显卡驱动由用户的系统提供。

Release 流程会调用 **Build CPU, OpenCL and CUDA Conda packages**，实际构建和测试三套包，并在新用户环境中同时安装验证；任一构建失败将阻止后续附件发布。该工作流也可手动运行，产物位于其 Artifacts 中。安装包不会自动上传 Bioconda。这个流程用本地源码路径验证配方，正式投稿材料仍使用固定 Release 源码 URL 和 SHA256。

安装测试检查版本及小样例哈希，普通无 GPU runner 测到的是 CPU 默认路径。真实设备测试需要手动运行 **Verify on a real GPU**，配置好自托管 GPU runner 的标签与运行环境。可选 `-R hmmer` 模式另需 HMMER。

GitHub 上生成材料不等于 Bioconda 审核通过；投稿后仍需处理其 CI 和维护者反馈。官方流程：https://bioconda.github.io/contributor/setup.html

## 如果发布工作流失败

先看 Actions 中失败步骤的日志。标签必须与项目版本一致；当前只能使用 `v0.1.0`。如果版本不一致，流程会报错并停止生成附件。

流程不会覆盖同名的现有 Release 附件。重跑时如果提示文件已存在，先确认现有文件是否完整。若需修改已经公开的源码，请增加版本并发布新 Release，不要挪动旧标签或替换旧源码包。

## 下次修改版本

统一修改 `VERSION`、`protlocus.h` 的 `PROTLOCUS_VERSION`、`recheck.c` 的 `RC_VERSION` 和 `tools/protlocus_genemodel.py` 的 `VERSION`，同时更新文档、man page 与发布说明。运行：

```bash
python3 tools/prepare_release.py --check
make clean
make -j2
make check
```

推送后，用对应的新标签创建 Release。自动流程会为新版本计算地址和校验值。

## 不通过 GitHub 的本地预演

在已提交代码的 Git 仓库中可以运行（将示例参数换成实际信息）：

```bash
python3 tools/prepare_release.py \
  --repository YOUR_ACCOUNT/YOUR_REPOSITORY \
  --maintainer YOUR_GITHUB_USERNAME \
  --output dist
```

只生成本地材料，不上传任何文件。输出目录已加入 `.gitignore`。重复预演应选择新的输出目录。

## 验证范围

本机通过 Linux amd64 容器实际构建后端；具体结果见 [验证记录](VALIDATION_zh.md)。普通容器与 GitHub 托管 runner 均没有可访问的真实 GPU，不把回退到 CPU 的样例或跳过 GPU 测试算作设备验证通过。

远程 Actions、Bioconda 官方构建和审核须在上传后完成；本次未发布任何远程内容。
