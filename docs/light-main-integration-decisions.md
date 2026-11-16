# Light 与主线整合决策

## 基线与当前状态

- 主线：`64322a9775af04c5cb3b7b144e1c239b17883c6d`，模块解耦与公共实现重构（#131）。
- 本地基线：`768ec3ff232aa0e549af699e1e3870096e38aa8c`。
- 整合分支：`codex/integrate-light-main-20261116`。
- 独立工作区：`/Users/guoyuqi/atrex-kernel-agent-integrate-main-20261116`。
- 已在整合分支保存附件竞态修复：`3b414197`。原工作区未修改。原工作区后续提交 `10926c14` 的附件权威库只追加、V0 身份固定修复也已纳入整合代码。
- 已执行 `git merge --no-commit --no-ff origin/main`。初始产生 172 个未解决路径；现已解决全部冲突并完成下述本地集成与安装包验证。

## 用户已确认的合并方向

采用主线公共模块边界：

| 模块 | 负责的能力 | 本地能力接入方式 |
| --- | --- | --- |
| `aka.agent` | Agent 调用、Session、进程与会话证据 | 接入持久 Session、完整对话与独立 token 审计、private/host 隔离 |
| `aka.evaluation` | 评测准备、结果解析、完整性和目标聚合 | 接入本地 Native/SOL 评测、完整统一回执、原始证据、ABBA 兼容层 |
| `aka.execution` | 执行、路由、Job 生命周期与传输 | 接入本地 Gateway 策略；Agate 统一 SDK-only，显式 SSH 保留独立传输 |
| `aka.experiment` | 测量身份、证据、去重、复用、重试 | 接入 Campaign 中央 Measurement Store，避免恢复每条 lineage 独立副本 |
| `aka.task` | 题目、源码结构与工作区契约 | 接入持久 Agent Git、可编辑 `kernel.py` 与可选 `impl/` |
| `aka.resources` | Prompt、Skill 与共享资源 | 搬迁本地新工作流资源，不恢复旧 Light 的 handoff/promotion 指令 |
| `aka.workflow.light` | Light 特有编排 | 保留 Campaign、Research Library、Submission、lineage、共享可见性与持久 Session 设计 |

公共模块的 Experiment 指评测执行和证据能力，不等于 Agent 写入 Research Library 的 Experiment 条目。后者继续属于 Light 研究工作流。

采用主线 `WorkflowLaunch`，使用 `--problem`、`--workspace-base`，不再要求 `--` 分隔，不恢复 `start_optimize.py` 和 `--op-dir` 转换链。

Light 不恢复 Framework Baseline、`episode-report`、handoff、promotion 或旧 Episode Journal 作为主流程依赖。根据最新决定，删除 Legacy 专用工作流、启动配置、安装包、资源与兼容分支；产品只保留 AKA Light，公共模块继续按职责复用。

保留本地 Fast/Performance/Ultra、中央 Research Library 版本与所有权、同步水位/异步可见性、accepted submission winner、现有八个可观测性工具。需要适配公共接口，不能仅批量替换 import。

## 用户已确认的运行策略

| 项目 | 最终选择 |
| --- | --- |
| 测量次数 | 默认 warmup 10、bench 20；提供启动选项，显式请求值优先 |
| 时间预算 | Light 统一可配置 `evaluation_step_timeout`，默认 60 秒；传入公共 Evaluation，覆盖 candidate/performance 默认。按 schedule 推导外层预算；不静默截短显式值 |
| 分片与锁频 | 每片 10 Shape、Ray Data、`shape_shards=1`。普通 Evaluate 使用完整 Reference + 选中片内 shape IDs；ABBA 暂用片内 Shape 重建 Reference，不传 shape IDs。只请求 `lock_clocks=true`，不指定 MHz |
| Agent/Review | Episode 与全部 Review 默认不设 active-time 上限；保留取消能力和 GPU/传输等待预算。Claim Reviewer 跟随所选 Agent sandbox：启用时用 Bubblewrap，未启用时在独立工作区原生运行 |
| 重试 | 保留本地策略。有 Job ID 时在预算内短轮询同一个 Job；无 ID 只有具备持久幂等保证才能重放。普通已确认终态基础设施失败/Job deadline 最多额外两次重提 |
| Gateway | Agate 全部通过 SDK，接入公共 Execution；不恢复 CLI/direct HTTP fallback。SSH 保留独立执行路径 |
| 验收与目标 | `submit-kernel` 检查同一 commit 的完整 Evaluate 正确性与可解析、完整性能结果。winner 按题目目标从 accepted submissions 中选取，不恢复 promotion/自动回滚 |
| 错误 | 吸收主线结构化错误与 SDK verdict 校验；行级成功不能覆盖 SDK 拒绝。完整原因归档，Agent 回执保留完整结果及诊断，不再做长度裁剪 |
| Shape 与源码 | 默认公开 Shape，`kernel.py` 可编辑、`impl/` 可选。请求隐藏但无法建立真实隐藏投影时明确拒绝，不允许静默暴露 |

## 初始源码与全部测量

启动只准备题目、初始源码、持久工作区和共同 Agent Git 根，**不强制执行 V0 Evaluate**。是否测量初始源码由 Agent 自主决定；没有测量时不创建虚假 baseline latency、speedup 或成功 Record。单线和 pooled search 都不得再依赖 measured V0 才能启动。

Campaign 只保留一个 `measurements/`。以下结果不论是否通过、是否最终提交，全部保留：

- Agent 主动执行的初始源码测量；
- 完整与子集 Evaluate；
- ABBA 的各分片、每次执行与原始响应；
- Profile、Dev、Command、Check、Disassemble 等测量或探针；
- 参数/准备错误、编译或正确性失败、超时和基础设施错误；
- 原始 Job 返回、公共 Execution 尝试与解析结果、Runtime 返回给 Agent 的回执。

普通请求复用中央已有结果时仍正常返回，不制造第二份物理测量。Runtime audit 保留每个调用者的请求和消费关系。失败证据仍可查询，但不会因归档而成为可复用的成功测量。

各 lineage 的 `records/` 仅按当前可见性投影；归档不受研究条目 Review、提交或 winner 资格控制。accepted 状态只影响 winner/Pareto/广播选择，不控制测量是否入库。共享初始 Git 根不代表已测量共同 V0。

公共 Task 仅保留 Light 所需的源码读取、初始建库、历史导出与同伴导入能力。移除仅供 Legacy 使用的 seal/promotion/restore 工作流接口；优化期间 Agent 自主管理 HEAD。

## 实施约束与验证范围

1. 不接受 Git 对目录重命名的所有猜测。例如把 `agent_git.py`、`baseline.py` 放到 `aka` 根目录并不是公共 owner 设计；须按职责放入公共模块或 Light 编排。
2. 先确定公共接口及新启动入口，再搬迁本地实现；不在最终分支保留第二套 `aka.light` 底层执行栈。
3. 不因为文本自动合并成功就认为行为正确。特别检查公共 Task 的 revision 操作是否会修改 Agent HEAD，公共 Experiment 是否会恢复 `.atrex_long_horizon` 私有状态路径，以及公共资源是否恢复旧 Prompt。
4. 保留附件竞态修复：中央不可变附件库不能被落后的 Catalog 快照清理；只裁剪 Agent 可见投影。
5. 决策确认后验证：新 CLI/WorkflowLaunch 启动；Native 与 SOL 输入；三种模式；单线、同步及异步共享；Direction/Experiment 版本与附件；Evaluate/submit winner；Session/Git 恢复；token 与可观测性；SDK 错误与幂等重试。
6. 修改仅位于独立整合分支；本次不推送、不更新沙箱，不修改原工作区。

## 已完成的整合

- `aka optimize` 由 `atrex-aka-light` 注册，依赖公共 `atrex-aka-framework`；删除 Legacy 专用 Workflow、包、启动配置及旧 Episode evidence reader/interpreter。
- `WorkflowLaunch` 只接受 Fast/Performance/Ultra，使用 `--problem` / `--workspace-base`。参考目录支持重复的 `--agent-reference-paths [NAME=]DIR`。
- Task 保留初始 Agent Git 建库、读取、历史导出及同伴导入；移除 Supervisor seal/promotion/restore/worktree 工作流方法。Episode 恢复不改变 Agent HEAD、分支、暂存区或未提交改动。
- Runtime 通过公共 Agent 的 Unix-socket ToolHost 提供能力，经公共 Experiment/Evaluation/Execution 完成测量；成功、失败、准备拒绝和 Env 观察均进入中央 Measurement Store。
- 完整 Evaluate 的 commit、scope、逐 Shape 结果与 submit-kernel 对齐；accepted submission 更新 Research Board。修复合并时无 baseline 的普通 Evaluate 被错误登记 baseline path 等接口衔接问题。
- Agate SDK 负责上传、提交、查询、取消和 Env；已接受 Job 在预算内短轮询。等待耗尽返回 `gateway_wait_timeout`，不再以泛化 `unknown` 指导 Agent；私有事实仍保留远端终态未确认的状态。
- Agent 默认无 active-time 上限；Bubblewrap 仅有 private/host。Claim Review 仍有独立证据工作区，但不强制启用 Bubblewrap；启用隔离时保留输入只读、输出可写。
- 更新启动 Skill、配置示例、工作流、Runtime 与隔离说明，移除强制测量 V0、旧启动分隔符及每 Episode 清空 scratch 等过时说法。

## 验证结果与范围

使用 Python 3.13，本地 31 项回归全部通过：

| 范围 | 测试数 | 覆盖 |
| --- | ---: | --- |
| 中央测量存储 | 4 | 成功、候选失败、准备拒绝、超时归档 |
| 测量配置和结果契约 | 7 | 可配置默认、参考目录、分片、ABBA、题目目标、SDK verdict |
| 初始源码与 Git | 4 | 无强制 V0 测量、公共初始 Git 根、源码契约 |
| Light-only 启动 | 5 | 新入口、包边界、无旧 Git 工作流、private/host |
| Runtime 集成 | 4 | 中央去重复用、失败审计、持久 WIP、Reviewer 随 sandbox 选择 |
| SDK 适配 | 4 | 上传/提交/短轮询/取消/Env、结构化拒绝、暂时查询失败恢复、明确等待超时 |
| 完整组合 Campaign | 3 | 三种模式、四种 Research 开关、单线及同步/异步多线、Evaluate/submit |

完整组合测试使用真实 Bootstrap、公共 owner、Session/ToolHost/Runtime、持久工作区与存储；Agent 模型、GPU 执行和 Claim Review 结论使用替身。Command 和工具 help 通过真实 `tools/sandbox.py` 子进程调用。

两个发行包的 sdist/wheel 构建成功，并在独立临时虚拟环境安装验证。新 CLI help、公共模块/资源加载，以及安装版完整 Evaluate → submit Campaign 测试通过。源码编译、静态未定义名称检查、安装脚本语法检查及 Git whitespace 检查通过。

这些验证不包含真实 Agate GPU 作业、模型 Provider、Linux Bubblewrap namespace 或 SSH 主机连接；没有启动或修改任何远端沙箱。代码已按用户确认的取舍完成整合，目前没有待用户决定的冲突。

## 后续本地更新：`fdb22d3e`

已按当前公共模块边界移植 Runtime 审计与可观测性更新，而非恢复原 `aka.light` 执行栈：

- Agent CLI 使用 `--kind evaluate`；内部 Execution 的 `run` 仍是私有传输选择。
- 所有 Runtime 工具使用 `operation/status/result` 统一回执。返回完整规范化结果，保留 Record ID 和路径；Dev/Command 输出不再裁剪为预览。
- Runtime 请求审计保存 Agent 请求、服务请求/返回及 Agent 实际收到的回执。Env 的 SDK 交换通过公共 Core 请求上下文采集；Wiki、Claim、Final Source 和专项数值 Review 也接入同一审计链。
- Env 观察继续写入中央 Measurement Store，保留合并分支“所有测量/观察入库”的设计，不采用旧分支“Env 无 Record”的说明。
- 可观测性区分最终逻辑调用状态与已恢复的物理尝试错误；Dashboard 支持调用详情、SDK 请求、服务返回、Agent 回执和 token/latency 曲线。
- 保留持久 Campaign/Session/Git、中央库及 accepted submission 语义。删除旧分支引入的 Baseline/验收恢复代码；Claim Review 仍跟随用户所选 sandbox。
- 组合测试发现并修复异步共享的快照竞态：Direction、Experiment 与共享排序读取同一版本，避免并发发布的新排序引用本次尚未导入的 Direction。

本次验证：101 项测试、18 个子测试通过，覆盖完整组合 Campaign、同步/异步共享、中央测量、SDK、三类 Review 审计、可观测性导出和 Dashboard；两份 Skill 格式校验、改动 Python 文件语法检查及 `git diff --check` 通过。Agent/GPU/Provider 使用离线替身，未进行实卡或远端沙箱验证。本次更新尚未提交。
