# `--reference-dir` —— atrex-bench 算子包格式（`agate eval --backend=atrex`）

**自定义算子走这条路。** 与 SOL 不同，题目（参考实现 + 输入生成 + shape 列表）**由你随请求上传**，
所以任何算子都能测，不受题库限制。

```bash
agate eval --gpu L20D --candidate my_kernel.py --reference-dir data/rms_norm
```

`--candidate` 是你要测的实现，`--reference-dir` 是判它对不对、按什么 shape 测的那套东西。

## 目录内容

调用方必须提供 `reference.py`、`input.py` 和 `shapes.json`。`metadata.json` 与 `roofline.json` 可选，
网关会为上游补齐缺省的 metadata 文件：

```
data/rms_norm/
├── reference.py     # 标准答案：class Model(nn.Module)
├── input.py         # 输入生成器：_make_inputs(**input_kwargs) -> dict[str, Tensor]
├── shapes.json      # 测哪些 shape
├── metadata.json    # 可选：算子元信息，缺省时由网关补齐
└── roofline.json    # 可选：有它才算得出 sol.pct
```

本文件下方给出所需文件和可选元信息的最小结构。有仓库权限时，还可以参考
[`examples/rms_norm/`](https://code.alibaba-inc.com/tre-infra/atrex-gpu-gateway/tree/master/examples/rms_norm)，
但它不是使用本 skill 的前置条件。

### `reference.py`

顶层必须有 `class Model(nn.Module)`，用 `Model(**init_kwargs)` 构造、`forward(**inputs)` 调用：

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, weight):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * weight
```

**你的 `--candidate` 也要暴露同名的 `class Model`**，签名一致。
候选、reference、输入生成器与 ABBA baseline 可使用普通 `input` 变量或参数名，
例如 `forward(self, *, input)`；内置交互函数 `input()` 仍不允许调用。

### `input.py`

顶层函数 `_make_inputs`，参数名对应 `shapes.json` 里的 `input_kwargs`，返回张量字典：

```python
import torch


def _make_inputs(token_count: int, hidden_size: int) -> dict[str, torch.Tensor]:
    x = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device='cuda') * 0.02
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device='cuda') * 0.02 + 1.0
    return {'x': x, 'weight': weight}
```

返回的 key 要对上 `Model.forward` 的参数名。领域约束（合法的 block 数、mask、cache 布局等）
都在这里构造——这是它是一段 Python 而不是一份声明的原因。

源码默认最多 **16 KiB＝16384 个 UTF-8 字节**，包含注释与空白，不是字符数或生成张量的大小。
部署方可用 `max_input_source_bytes` 单独调整，重启后生效；实际额度以
`GET /v1/env/{gpu}/capabilities` 的 `limits.max_input_source_bytes` 为准。
超限在排队前返回 `400 invalid_source`，错误指出 `reference.input_py` 的实际字节数与上限。
候选与 `reference.py` 默认各 128 KiB，使用独立的 `limits.max_source_bytes`，不会放宽输入生成器的额度。

### `shapes.json`

顶层是 dict，key 是字符串 shape id，value 分成构造参数和输入参数两半：

```json
{
  "0": {
    "init_kwargs": { "eps": 1e-06 },
    "input_kwargs": { "token_count": 2500, "hidden_size": 1024 }
  }
}
```

`init_kwargs` 喂给 `Model(**init_kwargs)`，`input_kwargs` 喂给 `_make_inputs(**input_kwargs)`。
多个 shape 就多加几个 key。

### `metadata.json`

算子元信息（category / dtype / 来源等）。**对调用方是可选的，但上游 gate 强制要求文件存在**，
所以网关永远会物化它——你写 `{}` 也行。

### `roofline.json`（可选）

带上它，每个 shape 的结果会多一个 `sol.pct`（对你自己那份 roofline 的达成率）：

```json
{ "shapes": { "0": { "SOL_time_ms": { "NVIDIA L20": 0.0123 } } } }
```

硬件名要对得上结果里的 `environment.gpu_name`。算不出来时回 `sol.pct: null` 并在
`sol.reason` 说明原因，不会回 0 冒充。

⚠️ 这里的 `sol` 是 **speed-of-light（性能上限达成率）**，与 `--backend=sol`（SOL-ExecBench 题库）
毫无关系，也与 `profile --level sol` 是同一个词的第三种用法。

## 结果怎么读

回 `compile` / `correctness` / `performance` 三段。判成败**看 `passed`，不要看 `error`**——
编译失败时上游会把 `error` 留成 `None`。

每个 shape 还带：

- `timing_stats.cv`——离散度。cv 大说明这次数字不该拿来比。
  **cv 为 `null` 表示只采了一个样、离散度无从判断**，不是「很稳定」；`runs` 告诉你实际测了几次。
- `environment.clock_sm_pct_of_max` / `clock_throttle_reasons`——实测频率，判断这次测量可不可信。

## 该用这个 backend 吗

| 你的情况 | 用哪个 |
|---|---|
| 自己的算子，自带参考实现 | `--backend=atrex`（本文件，默认值） |
| 给 SOL 235 题里已有的某道题写实现 | `--backend=sol`，见 [solution-json.md](solution-json.md) |
