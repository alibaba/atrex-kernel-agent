# wiki-gate

## 角色

接入门禁:来自 trace/session 的 wiki 总结进入 `kernel_wiki/records/` 的唯一入口。
同时也是 trace 批量扫描时更新 counters 的执行器——反馈不走实时上报,而是由 trace 分析批量触发。

## 两步接口

gate 是**纯确定性工具**,不含任何模型调用。语义判断由调用方(agent 或 trace 分析脚本)自行完成。

### Step 1:`--match`(返回候选,不做决策)

```bash
python3 scripts/gate.py --match --input /path/to/incoming.json
```

输出(stdout,单个 JSON):
```jsonc
{
  "status": "ok",
  "incoming": { "id", "type", "change", "mechanism", "goal", "gain_direction", "gain_pct", "shape_contract", ... },
  "candidates_count": 3,
  "candidates": [ { 同上字段 } × 全部同 scope 记录 ]
}
```

调用方(agent)读取 candidates,自行判断:无匹配 → insert;有匹配且方向一致 → confirm;有匹配但方向矛盾 → conflict。

### Step 2:`--commit`(执行决策)

```bash
python3 scripts/gate.py --commit insert   --input incoming.json
python3 scripts/gate.py --commit confirm  --input incoming.json --target <existing_record_id>
python3 scripts/gate.py --commit conflict --input incoming.json --target <existing_record_id>
```

| action | 效果 |
|---|---|
| `insert` | 通过校验后写入 records/,**并把条目追加进 `records/index.json`**(否则新记录对下一次 `--match` 不可见,主库第 7 道门也会 FAIL);stdout 打印路径 |
| `confirm` | 原记录 `worth.track.counters.verified_effective` +1(就地改写),用 `worth.track.confirm_keys` 去重,同一条来源记录重复上报不会重复计数 |
| `conflict` | 完整冲突信息写入 `kernel_wiki/conflicts/`,原记录不动 |

`insert` 会**拒绝已存在的 id**(同路径或索引中任何位置),提示改用 `confirm` —— 覆盖写入等于为了放进一条以为是新的记录而删掉一条已有的。

### `insert` 的校验范围

门禁自称是进入 `records/` 的唯一入口,所以它跑的是**主库同一套记录级校验**,而非只有 schema:

- JSON-Schema(clean-1.3)
- 既定事实(见下)
- 主库的 4 道记录级门,直接复用 `tools/check_kernel_wiki.py`:
  `anonymization` / `raw-isolation` / `self-contained` / `no-cross-reference`

余下 4 道(`ids` / `coverage` / `relations` / `index`)是全库性质的,由 `insert` 的 id 冲突检查与每批之后的 `--full` 覆盖。

若 `tools/check_kernel_wiki.py` 无法导入,门禁**拒收而非跳过** —— 静默少跑 4 道门正是它要防的事。

### 候选池的 scope 过滤

`vendor` / `arch` / `dsl` / `operator_family` 四项硬过滤。`dsl` 的 `any` 意为「与 DSL 无关」,因此**双向匹配**:`dsl=triton` 的 incoming 能看到 `any` 候选,`dsl=any` 的 incoming 也能看到全部 DSL 的候选。单向版本曾让一条 `any` 记录拿到 0 候选,而主库里有一条 `dsl=triton` 的近乎逐字重复(+15.5% 对 +15.78%),差点被当新知识插入。

### 反面教材的准入:必须是既定事实

`insert` 一条 `anti-strategy` 时,gate **拒收**下列任一情况:

- `payload.established_fact` 缺失
- `established_fact.condition` 四项(`sm_arch` / `shape_regime` / `dtype` / `toolchain`)全为空 —— **「在这个算子上」不是条件**
- `established_fact.mechanism` 短于 40 字符,或整句只是测量结果(「测了没涨」「3 种都 flat」)
- `verdict` 为 `unstable` / `unknown` —— 这两个值已从 schema enum 移除,一次没有结论的运行不是负面知识

被拒的记录留在挖掘 skill 自己的暂存区,不删除:等它补上条件与机制、或被多次独立复现之后,可以重新过 gate。

判据全文、五类算机制 / 六类不算机制、边界判例见
`skills/wiki-gate/references/established-fact-criteria.md`;
同一套判据由 `tools/check_kernel_wiki.py` 的 `established-fact` 门在主库侧强制。

## 触发时机

- **新 wiki 入库**:opt-trace-mining / session-trace-mining 产出新记录后,由 agent 调 `--match` → 自行判断 → 调 `--commit`
- **反馈回写**:定期 trace 扫描发现某条 wiki 被 agent 采纳且保留 → 调 `--commit confirm`

不存在「agent 实时上报」的路径。所有反馈来自 trace 文件的事实记录。

## 方向矛盾的定义

- 一条 strategy + 一条 anti-strategy(或反之)
- 两条 strategy 但 gain.pct 一正一负

**数字大小差异不算矛盾**(shape / 硬件条件不同的正常偏差)。

## 冲突文件

写入 `kernel_wiki/conflicts/<timestamp>_<id>.json`,保留完整的新 wiki 信息(含 shape_contract、bottleneck、observed_symptom),方便后续排查差异原因。`resolution` 字段初始为 null,人工验证后填入。

## 设计约束

- **纯确定性**:gate 不做语义判断、不调模型;硬过滤 + 特征提取 + 文件 IO
- **幂等**:同一条重复提交,confirm 的 increment 事件 key 包含双方 id,重复写入无效果
- **不改原记录内容**:gate 不修改已有 record 的 payload / evidence;只通过 events.jsonl 更新 counters
- **冲突不阻塞**:标记后正常退出(exit 0);冲突文件是给人看的队列,不是错误
- **全量候选**:--match 返回该 scope 下所有记录,不截断;调用方可分批处理

## 依赖

- Python 标准库
- jsonschema(可选,缺失时仅做版本号校验)
- 不依赖任何模型 API
