# atrex eval 的 solution manifest

`POST /v1/jobs/eval` 的 `candidate` 接受三种互斥表示：原有 Python 源码字符串、
原有 `{archive, entry_point}` OSS 源码包引用、版本化 manifest 对象。CLI 使用
`--manifest FILE.json`，与 `--candidate`、`--candidate-entry`、`--config` 互斥；
仍需 `--reference-dir`。这是 atrex eval 的输入格式，和 `--backend sol --solution` 的
SOL-ExecBench `solution.json` 是不同契约。`profile` HTTP 同样接受 manifest；compile 和旧 `/v1/evals` 暂不接受 manifest。

## v1 字段

```json
{
  "schema_version": 1,
  "name": "add-one",
  "version": "1",
  "entry_point": "impl/entry.py::Model",
  "source": {
    "type": "inline",
    "files": {
      "impl/__init__.py": "",
      "impl/entry.py": "from .helper import Model\n",
      "impl/helper.py": "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x): return x + 1\n"
    }
  },
  "dependencies": [
    {"id": "torch", "kind": "pip", "requirement": "torch>=2"}
  ],
  "targets": {"hardware": ["H20"], "num_gpus": 1, "shape_ids": ["0", "1"]}
}
```

- 必填版本、名称、实现版本、源码和入口；未知字段拒绝。Python 入口为相对文件路径加
  `::Model`，支持包内相对导入；CUDA 入口使用下述对象声明，由 worker 编译并生成适配层。
  不会在网关进程中 import 或编译用户代码。
- `source.type=inline` 时 `files` 是路径到 UTF-8 源码的映射。每个 Python 文件沿用源码限额和
  AST 检查，所有文件合计最多 4 MiB、2000 个文件，路径不能绝对、穿越或冲突。
- `source.type=oss_archive` 时提供 `archive`（对应 `oss_files[].path`）、压缩包 `sha256`，
  以及 `files`（完整文件清单，路径到 UTF-8 文件字节的 SHA256）。源代码仍经 OSS 直达 worker；
  worker 受控解包后核对全部文件和摘要，再沿用源码校验与入口适配。二进制请作为独立附件。
- 每条依赖有唯一 `id` 和可选 `depends_on`，引用其他依赖 ID；缺失节点、环和重复 ID 提交即拒绝。
  `pip` 用 `requirement` 声明包，与顶层 `requirements` 按文本去重合并，经现有依赖验证和安装流程，
  `deps_mode` 仍默认 `freeze_installed`，不自动更改解析策略。
- `system` 用 `name`、`version` 描述镜像中的系统库／工具链先决条件，网关不安装或验证系统包。
  `artifact`／`weights` 用 `path`、`sha256` 引用独立 OSS 附件，worker 验摘要并将其映射到
  源码树中的同名路径，便于通过 `__file__` 定位。附件内容不会进入查询元数据。
- Python 入口的 `build` 依赖支持 `execution=entry_point`，声明 `toolchain`、`inputs`、`outputs` 和依赖边。
  `inputs` 引用源码、声明附件或前序构建输出；使用其他节点的产物必须有依赖边。
  `outputs` 是描述构建产物的逻辑名称，不是网关执行的写盘指令或自动收集配置。
  实际编译／代码生成仍由候选入口完成；不会执行 manifest 中的 shell 命令，也没有新的 DAG 调度器。
  pip、系统和附件准备不能依赖尚未执行的入口构建。
  CUDA 类型入口的实际构建独立配置在 `entry_point.build`，不能同时声明这类入口内构建依赖。
- `git` 声明版本化仓库，由 worker 自动准备；用法见下节，不局限于特定 GPU 厂商或仓库。
- 可选 `targets.hardware` 必须覆盖请求的全部硬件别名，`num_gpus` 须与请求一致，
  `shape_ids` 必须覆盖 reference 的全部 shape。`compile_targets` 描述入口的编译目标，
  不注入编译参数，也不代表 worker 已验证支持该架构。

## 校验与元数据

`GET /v1/jobs/eval/solution-schema` 返回模型生成的 JSON Schema。
`POST /v1/jobs/eval/validate` 接受完整 eval 请求，复用提交时的结构与源码校验，返回
`valid`、`scope=declaration_and_inline_source` 和 `solution`，不提交作业、不下载 OSS。
结构错误返回 `422 request_validation_failed`，语义／源码错误返回 `400 invalid_source`，
均在 `details.validation_errors` 中定位。校验接口不保证 OSS 上传凭据、目标资源或环境已就绪。

提交时能检查源码清单、路径、依赖环、声明约束和已提供的 Python 内容；内联入口明确未声明或
导入 `Model` 时拒绝。OSS 内部真实文件、摘要和动态 Python 符号只能在执行侧解包／加载后确认，
不能将提交成功当作运行时入口存在的证明。源码包／附件摘要错误在 SDK 启动前作为
`source_archive_invalid` 失败，不重试。

查询 `GET /v1/jobs/{job_id}` 的 `solution` 可读取无源码的声明、文件摘要、依赖边和拓扑顺序。
含 URL 的 pip 依赖在查询和校验结果中替换为 `sha256:<摘要>`，不回显 URL 或临时凭据。
同一份信息位于持久化 `ExecutionSpec.metadata.solution`，随 attempt 的执行规格保留；
旧字符串／源码包也投影为最小源码描述，旧源码包未声明的文件清单和摘要保持未知。
`declaration_sha256` 只对 manifest 声明做指纹，不是可直接复用产物的缓存 key：缓存仍须结合
实际硬件、工具链、安装结果、环境参数和评测器版本判断。第一版没有缓存命中或共享制品承诺。
manifest 目前串行执行，自动 shape 并行会回退串行，显式 `--shape-shards` 大于 1 会拒绝。
ABBA 的候选与 baseline 均可使用 manifest，各自准备源码、依赖和编译产物；评测器本身须支持
`eval_mode=abba`，旧镜像的评测器不支持时需用 `--atrex-bench-version` 指定兼容版本。

## 自动准备仓库依赖

在 `dependencies` 中声明仓库和版本，例如 CUTLASS：

```json
{
  "id": "cutlass",
  "kind": "git",
  "repository": "https://github.com/NVIDIA/cutlass.git",
  "ref": "v3.9.2",
  "include_dirs": ["include", "tools/util/include"]
}
```

同一契约可声明 CK 或其他仓库，例如
`{"id":"ck","kind":"git","repository":"https://github.com/ROCm/composable_kernel.git","ref":"rocm-6.3.0","include_dirs":["include"]}`。
可同时声明多个仓库，按依赖拓扑顺序准备。仓库依赖不表示执行环境已经支持该仓库的编译工具链；
原生入口目前为 CUDA，CK 的 HIP 编译评测不在该入口的支持范围内。
现有 pip 安装和附件准备发生在仓库准备之前，不能反向依赖 Git 节点。

- `repository` 必须是 worker 可访问的 HTTPS 地址，不能包含账号、密码、query 或 fragment。
  使用 Git 的 HTTPS 传输，不读取全局 Git 配置或 credential helper；当前不提供私有仓库鉴权参数。
  worker 须安装 Git。网关只验证声明，不在 HTTP 进程中联网拉取仓库。
- 可选 `mirrors` 是按优先级排列的 HTTPS 镜像地址列表，与 `repository` 使用相同 URL 校验。
  worker 先尝试镜像，镜像不可访问、缺少版本或不匹配锁定 commit 时，继续下一镜像，最后尝试原仓库。
  原仓库拉取遇到网络／工具基础设施错误时最多尝试三次，间隔 1／2 秒；版本或认证错误不重试。
  所有来源使用相同 `ref`／`commit`；建议同时锁定 `commit`。未声明镜像时直接使用原仓库。
  镜像地址由调用方配置，不内置组织私有地址，也不自动传递本机凭据。
  示例生成器支持 `--git-mirror "$REPOSITORY_MIRROR_URL"`（可重复）；内源镜像须允许 worker 直接读取，否则会回退。
- `ref` 是 tag 名、`refs/tags/<tag>` 或完整 40 位 commit；不接受分支或任意 refspec。
  tag 会解析为实际 commit，可通过可选 `commit` 字段要求与固定的 40 位 SHA 完全一致。
  不存在的版本、锁定不匹配和缺失 include 目录在编译前失败。
- `include_dirs` 是仓库内相对目录，允许 `.` 表示仓库根目录；CUDA 类型入口自动把这些目录加入编译。
  用户只需在 `.cu` 中正常 `#include`，无需把第三方头文件塞进 `source.files`。
- 仓库准备在本次 attempt 源码树的 `.agate_dependencies/<id>`，该目录保留给 agate。
  Python 类型入口也可相对于自己的源码树引用它；原有源码校验不变。
  仓库文件不占用候选源码的 4 MiB 限额；每次作业独立拉取，不使用共享可变目录或跨作业缓存。
- 不执行仓库安装／构建脚本，不自动初始化 submodule 或下载 Git LFS 对象。
  需要其他仓库时分别声明依赖。每条 Git 命令最多 300 秒，整体仍受作业执行超时限制。
- 结果的 `prepared_dependencies.candidate` 记录每个仓库的声明版本、实际 commit、逻辑目录和 include 目录；`fetched_from` 记录实际来源，
  `mirror_failures` 记录回退来源及错误类别（不包含远端原始诊断或凭据）。
  `solution.dependencies` 保留提交声明；静态校验通过不代表版本已经存在或能从 worker 下载。
  声明／版本问题为 `validation/repository_invalid`，网络或 Git 工具不可用为
  `infra/repository_unavailable`，不会回退到镜像内其他版本。

随包的 `cuda-cutlass` 示例仅提交一个使用 CUTLASS 的 `.cu`，依赖由 agate 拉取：

```bash
python make_manifest.py cuda-cutlass --gpu H20 --output solution.json
agate eval --gpu H20 --manifest solution.json --reference-dir reference
```

## 原生 CUDA 候选，不提交 Python

`source.files` 可以只有 `.cu` 和所需的原生源文件／头文件。把 `entry_point` 设置为以下对象：

```json
{
  "type": "cuda",
  "symbol": "launch_add_one",
  "build": {
    "toolchain": "torch_extension",
    "sources": ["impl/kernel.cu"],
    "cflags": ["-O3"],
    "cuda_cflags": ["-O3"]
  },
  "inputs": [{"kind": "tensor", "name": "x", "dtype": "float32"}],
  "outputs": [{"name": "out", "like": "x"}],
  "arguments": [{"ref": "x"}, {"ref": "out"}, {"numel": "x"}]
}
```

对应 `.cu` 中的 host launcher 签名如下，函数内自行选择 grid／block／共享内存并启动 kernel：

```cpp
extern "C" void launch_add_one(const void* x, void* out, int64_t n, cudaStream_t stream);
```

- `build.sources` 是完整源码清单中的 `.cu`／`.cpp`／`.cc`／`.cxx` 文件，至少包含一个 `.cu`；
  `include_dirs` 是源码树内的相对目录，`cflags`、`cuda_cflags`、`ldflags` 是编译参数数组。
  参数项不接受空白或 shell 元字符；不接受 shell 构建脚本。工具链固定为 `torch_extension`，
  worker 镜像须具备兼容的 PyTorch、CUDA toolkit、C++ 编译器和 ninja；系统库不会自动安装。
- `inputs` 按 `Model.forward` 参数声明名称和类型，名称必须与 reference 输入一致。Tensor 使用
  `kind=tensor`，支持 `float16`、`bfloat16`、`float32`、`float64`、`int8`、`uint8`、`int32`、`int64`、`bool`；
  运行时要求 contiguous、声明的 dtype 和同一 CUDA device，不隐式复制或转换输入。
  标量使用 `kind=scalar`，支持 `int64`、`float32`、`float64`、`bool`。第一版仅支持空 `init_kwargs`。
- `outputs` 使用 `like` 指定设备和默认 shape／dtype 的来源 Tensor，可用 `dtype` 覆盖输出类型；
  可选 `shape` 数组的每一维为非负整数或 `{"input":"x","dim":0}`，不执行任意表达式。
  输出是新分配的 contiguous Tensor；一个输出直接返回，多个输出按声明顺序返回 tuple。
  第一版不声明原地修改、输出别名或任意 Python 容器返回值。
- `arguments` 确定 launcher 的参数顺序：`{"ref":"x"}` 引用输入或输出，
  `{"numel":"x"}` 和 `{"size":"x","dim":0}` 传入 `int64_t` 元素数或维度。
  输入 Tensor 传 `const void*`，输出传 `void*`，标量按声明映射到 `int64_t`／`float`／`double`／`bool`；
  最后一个参数固定追加 `cudaStream_t`，使用输入设备当前的 PyTorch CUDA stream。
  launcher 必须使用 C linkage 并遵守该 ABI，将指针转换为声明的 dtype；网关不推断原生函数签名。
- worker 先验源码／附件摘要，再生成 C++ 绑定并编译本次 attempt 独立的扩展；生成的 `Model`
  在评测进程中只加载产物、调用扩展，ABBA 的新进程同样复用本次产物。编译计入作业总预算，
  不计入 kernel 测时；编译或扩展加载失败返回 `code/native_build_failed`，不自动重试。
  编译原生代码不提供 sandbox；依赖既有 worker 隔离，不建立跨作业共享构建缓存。

从随包示例目录执行，候选包内只有 `impl/kernel.cu`：

```bash
python make_manifest.py cuda-native --gpu H20 --output solution.json
agate eval --gpu H20 --manifest solution.json --reference-dir reference

python make_manifest.py cuda-native --gpu H20 --archive --output solution.json
agate eval --gpu H20 --manifest solution.json --reference-dir reference \
  --oss-file archives/solution.zip=solution.zip
```

reference 基准仍使用现有 Python 输入生成与参考实现契约；无需 Python 的是候选实现。

## 可运行的多文件示例

随包目录 [assets/eval-manifest](../assets/eval-manifest/) 提供四份 add-one 候选及共同 reference：
Triton 的入口／kernel 分文件，CUDA 的 Python 入口编译独立的 C++ 绑定和 `.cu`，
`cuda-native` 只提交 `.cu`，由 manifest 声明构建和调用方式，`cuda-cutlass` 进一步声明版本化 Git 依赖。
CUDA 示例限定 CUDA GPU、float32 输入和具备匹配 PyTorch／CUDA／C++ 工具链的镜像。
下面命令从该示例目录运行，硬件名按实际获准资源选择。

```bash
python make_manifest.py triton --gpu H20 --output triton.json
agate eval --gpu H20 --manifest triton.json --reference-dir reference

python make_manifest.py cuda --gpu H20 --archive --output cuda.json
agate eval --gpu H20 --manifest cuda.json --reference-dir reference \
  --oss-file archives/solution.zip=cuda.zip
```

SDK 继续使用已有 builder，传 `.json` 文件或内存中的 manifest 映射：

```python
import os
from atrex_gateway_client import AkSkAuth, Client, build_eval_request

client = Client(os.environ["AGATE_URL"], auth=AkSkAuth(os.environ["AGATE_AK"], os.environ["AGATE_SK"]))
request = build_eval_request("triton.json", "reference", "H20", shape_shards=1)
print(client.validate_eval(request))
accepted = client.submit_job("eval", request)
print(client.wait(accepted["job_id"]))
```

迁移单文件时，把源码放进 `source.files={"kernel.py": source}`，入口设为
`kernel.py::Model`，补齐 `schema_version`、`name` 和 `version`，用该对象替换 `candidate`。
reference、options、env_vars 和轮询／取消方式不变。无需迁移的调用方继续传原 `.py`。
