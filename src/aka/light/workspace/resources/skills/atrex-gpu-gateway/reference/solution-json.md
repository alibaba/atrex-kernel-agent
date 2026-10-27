# `solution.json` —— SOL-ExecBench 候选解格式（`agate eval --backend=sol`）

**这个文件装的就是你自己写的算子。** `sources[].content` 是内联的源码，`spec.entry_point`
指向你写的入口函数。「上传 solution」不是「上传别人的解」。

网关**逐字**转交上游：`agate eval --backend=sol --solution f.json` 对文件不做任何改写，
所以一份现成的上游 solution.json 可以原样提交。

## 最小骨架

```json
{
  "name": "my-rmsnorm-v3",
  "definition": "001_rms_norm",
  "author": "daocheng",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["B200"],
    "entry_point": "kernel.py::rms_norm"
  },
  "sources": [
    { "path": "kernel.py", "content": "import triton\n...你的 kernel 全文..." }
  ]
}
```

提交：

```bash
agate eval --backend=sol --solution solution.json --gpu L20N
```

## 字段逐条

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | ✅ | 这次提交的名字，随便起，非空即可。 |
| `definition` | 条件必填 | 题目名：镜像题库路径必填；带 `--problem` 时可由 `definition.json.name` 派生。建议始终保留，让文件也能被上游工具直接使用。 |
| `author` | ✅ | 上游强制非空。漏了会被上游拒掉（`author Field required`），不是网关的锅。 |
| `spec.entry_point` | ✅ | `文件名.py::函数名`，必须能在 `sources` 里解析到。这是 SOL 取代 atrex「隐式 `class Model`」契约的方式。 |
| `spec.target_hardware` | ✅ | 上游枚举**只有** `B200` / `LOCAL`，装不下我们的集群名。**填了也不参与选簇**——选簇一律看 `--gpu`。照抄 `["B200"]` 即可。 |
| `sources` | ✅ | 至少一个 `{path, content}`。`content` 是**内联源码字符串**，不是路径。 |
| `spec.languages` | | `pytorch` / `triton` / `cute_dsl` / `cutile` / `cudnn` / `cublas` / `cutlass` / `cuda_cpp` 等。 |
| `spec.dependencies` | | 上游的依赖声明。要装 pip 包请用 `-r requirements.txt`。 |
| `spec.destination_passing_style` | | 默认 `true`：输出预分配、作为尾部参数传入。与 atrex 侧默认相反，别照搬。 |
| `spec.binding` | | C++/CUDA 的 binding 风味，上游默认 `torch`。 |
| `spec.compile_options` | | 如 `{"cuda_cflags": [...], "ld_flags": [...]}`。 |
| `description` | | 自由文本。 |

## 多文件

`sources` 可以有多个元素，也可以放非 Python 文件（`.cu` / `.cpp` / `.h`）：

```json
"sources": [
  { "path": "kernel.cu",  "content": "__global__ void rms(...) { ... }" },
  { "path": "binding.cpp", "content": "#include <torch/extension.h>\n..." },
  { "path": "run.py",     "content": "from .binding import rms\ndef run(x, w): ..." }
]
```

`entry_point` 指向其中的 Python 入口（`run.py::run`）。路径必须是相对路径，不能有 `..`。

⚠️ **`content` 是 JSON 字符串**，源码里的换行、引号、反斜杠都要转义。手写容易出错，建议用脚本生成：

```python
import json, pathlib
srcs = [{"path": p.name, "content": p.read_text()} for p in pathlib.Path("mykernel").glob("*.py")]
doc = {"name": "my-v3", "definition": "001_rms_norm", "author": "me",
       "spec": {"languages": ["triton"], "target_hardware": ["B200"],
                "entry_point": "kernel.py::rms_norm"},
       "sources": srcs}
pathlib.Path("solution.json").write_text(json.dumps(doc, indent=2))
```

## 题目从哪来

有两种来源：

1. **镜像题库**：不带 `--problem`，`definition` 是内置 235 题之一。
2. **调用方自定义题**：带 `--problem DIR`，目录包含上游原生的 `definition.json + workload.jsonl`，
   完全不查镜像题库。完整、无需仓库权限的例子见 [custom-sol-problem.md](custom-sol-problem.md)。

镜像题库路径的规则：

- **提交期校验不了**：网关手上没有数据集，题目名写错是**运行时**的结构化错误
  （`validation/no_such_problem`，会指名是哪个题），不是 400。
- `--subset L1|L2|Quant|FlashInfer-Bench` 可以跳过 pod 内的题目搜索。不给就在四个 subset 里搜；
  0 个或多于 1 个匹配会明确报错，不是含糊的 infra 失败。
- 题数分布：L1=94、L2=82、Quant=33、FlashInfer-Bench=26。

## 会咬人的几处

- **`speedup` 默认恒为 0**。上游 `benchmark_reference` 默认 false，不测参考实现，
  于是 `reference_latency_ms` 是 0.0、比值除出来就是 0。要有意义的 speedup 必须显式
  `--benchmark-reference`，代价约 2× 运行时间。**别把默认的 0 读成「没有加速」。**
- **`--workload-timeout` 是每个 workload 的预算**（= 上游的 `--timeout`，默认 600s），
  `--job-timeout` 是整个作业的预算（默认 1800s）。一道题是多个 workload 串跑，别混。
- **`REWARD_HACK` 不判作业失败**。它是上游的一种判定，会作为证据回传，自己看
  `summary.reward_hack`。
- **结果里出现 `problem_patched: ["hf_id"]`** 表示网关改过题面才跑得起来：镜像里
  FlashInfer-Bench 那 26 道题的 `hf_id` 是空串、上游自己的模型又拒空串（源头在数据集发布链路，
  不是你的 solution 有问题）。网关只把这一个溯源字段置 null、只对本作业的副本生效，不参与任何判定。
  健康的 209 题不会出现这个 key。

## 该用这个 backend 吗

| 你的情况 | 用哪个 |
|---|---|
| 给题库里已有的某道题写了更快的实现 | `--backend=sol`（本文件） |
| 自己的算子，已有 atrex-bench reference 目录 | `--backend=atrex`，见 [reference-dir.md](reference-dir.md) |
| 自己的算子，希望使用 SOL 的 definition/workload/solution 契约 | `--backend=sol --problem DIR`，见 [custom-sol-problem.md](custom-sol-problem.md) |
