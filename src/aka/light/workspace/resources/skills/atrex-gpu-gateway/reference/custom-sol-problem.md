# 自定义 SOL 题：无需仓库权限的完整示例

当用户要自己出一道 SOL 题、把现有算子转换成 SOL 格式，或验证自定义题链路时读取本文件。
下面的内容完整包含在 skill 包内，只需要本机有 `python3`、`agate` 和网关鉴权。

## 一键生成示例

把下面代码保存为 `create_custom_sol_smoke.py` 并执行。它会在当前目录生成
`custom-scale2-smoke/`，其中包含上游原生的 `definition.json`、`workload.jsonl` 和
`solution.json`：

```python
from pathlib import Path
import json


root = Path("custom-scale2-smoke")
root.mkdir(exist_ok=True)

definition = {
    "name": "custom-scale2-smokev1",
    "op_type": "elementwise",
    "description": "Multiply a float32 vector by two.",
    "axes": {"N": {"type": "var"}},
    "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
    "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
    "reference": "import torch\n\n\ndef run(x):\n    return x * 2.0\n",
}

workloads = [
    {
        "axes": {"N": 4096},
        "inputs": {"x": {"type": "random"}},
        "uuid": "custom-scale2-4096",
    },
    {
        "axes": {"N": 1048576},
        "inputs": {"x": {"type": "random"}},
        "uuid": "custom-scale2-1m",
        "tolerance": {"max_atol": 0.001, "max_rtol": 0.001},
    },
]

solution = {
    "name": "custom-scale2-pytorch",
    "definition": definition["name"],
    "author": "agate-smoke-test",
    "description": "Minimal PyTorch candidate for the custom-problem smoke test.",
    "spec": {
        "languages": ["pytorch"],
        "target_hardware": ["LOCAL"],
        "entry_point": "kernel.py::solve",
        "destination_passing_style": False,
    },
    "sources": [{
        "path": "kernel.py",
        "content": "def solve(x):\n    return x * 2.0\n",
    }],
}

(root / "definition.json").write_text(
    json.dumps(definition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(root / "workload.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in workloads),
    encoding="utf-8")
(root / "solution.json").write_text(
    json.dumps(solution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(root.resolve())
```

```bash
python3 create_custom_sol_smoke.py
agate eval --backend sol \
  --solution custom-scale2-smoke/solution.json \
  --problem custom-scale2-smoke \
  --gpu L20N
```

上述命令默认使用正式环境。同一个题目目录也能交给上游本地工具：

```bash
sol-execbench custom-scale2-smoke \
  --solution custom-scale2-smoke/solution.json --json
```

## 验收标准

以下条件必须同时满足：

- 作业 `status` 是 `succeeded`。
- `result.summary.passed == result.summary.total`。
- `problem_source` 是 `inline`。
- `problem_digest` 以 `sha256:` 开头；同一份题面重复提交时 digest 相同。

自定义题的 `PASSED` 只说明候选解与调用方自带 reference 一致，不代表通过官方 235 题，不能把两者
混在同一个成绩统计中。

## 从现有算子转换

转换时按下表映射：

| 现有算子信息 | SOL 文件 |
|---|---|
| reference 实现 | `definition.json.reference` 的顶层 `run()` |
| shape 变量 | `definition.json.axes` |
| 输入名、shape、dtype | `definition.json.inputs` |
| 输出名、shape、dtype | `definition.json.outputs` |
| 每组真实 shape 与输入生成方式 | `workload.jsonl` 的一行 |
| 候选实现与入口函数 | `solution.json.sources[]` 与 `spec.entry_point` |

`definition.inputs` 的序列化顺序、`reference.run()` 的形参顺序、候选入口函数的形参顺序必须一致。
例如 inputs 按 `x, weight` 写，两个函数也必须接收 `(x, weight)`。题名必须匹配
`custom-<operator>-<tag>`，全小写且 tag 至少 4 个字符。

需要保持特殊输入分布时，在 reference 中增加 `generate_inputs(axes_and_scalars, device)`，设置
`custom_inputs_entrypoint`，并把对应 workload inputs 的 `type` 写成 `custom`。一个 workload 不要
混用 `custom` 与其他输入类型。
