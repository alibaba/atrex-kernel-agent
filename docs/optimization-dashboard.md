# 优化进展展示插件设计

可选的 `aka.dashboard` Core 插件通过独立 `Startup` 启动本地只读页面，展示优化阶段、实验结果和 token 消耗。Dashboard 注入 `observation_source` 服务，只消费统一观测契约；默认由 `aka.observation.long_horizon` 适配插件读取优化工作空间，一个页面可以观察多个框架 campaign。通过独立 launch profile 启用，安装和运行优化器时无需加载 UI。

采用 Python 标准库 HTTP 服务、原生 HTML/CSS/JavaScript 和两秒轮询。数据源负责原生记录的版本校验、语义转换和阶段定义，页面按契约动态展示阶段。运行配置、流式用量与全流程 receipt 仍属于后续扩展；UI 明确展示已结算、待结算和不可用，缺失预算不生成百分比。

## 现有能力

当前 Core 可以组装独立插件，由 Bootstrap 调用 profile 指定的 `Startup.run(Invocation) -> int`，并在调用结束后释放组合。UI 可以使用这个入口，无需新建一套插件框架。[Core 接口](plugin-framework.md)和[启动方式](application-plugin-migration.md)规定了具体契约。

仓库还有用于 Agent 工具和 Skill 的 `plugins/*/plugin.json` 与 `plugin_runtime/`。展示服务和数据源采用 Core 插件；HTTP 服务与资源由展示插件管理，优化文件协议仅由适配插件解释。以下文件表属于默认 long-horizon 适配器，不是所有 workflow 必须实现的磁盘布局。

| 已有数据 | 可以展示的内容 | 当前边界 |
| --- | --- | --- |
| `memory/live.json` | 活动 episode、目标版本、实验数量、最近实验、最近更新时间 | 在 journal 写入后刷新；属于非 canonical 进度 |
| `.atrex_long_horizon/active_episode.json` | episode 模式、工作树、准备、探索、验证、晋升等阶段 | 进程终止后可能仍然保留，不能单独证明任务正在运行 |
| `.atrex_long_horizon/state.json` | episode 结果计数、已结算 tokens、去重 receipt | 主要在 Agent invocation 返回后结算；统计 episode 主 Agent |
| `.atrex_long_horizon/episodes/eNNNN/attempt.json` | 接受或拒绝、原因、验证结果、episode tokens | 终态归档；和 state、telemetry 存在重复信息 |
| `.atrex_long_horizon/episodes/eNNNN/telemetry.summary.json` | 输入、输出、缓存、阶段用量、观测质量、原因码 | 通常在 episode 结束后生成；可能为 partial 或 unavailable |
| `memory/v<N>.json` | canonical 版本性能、正确性、优化摘要 | 拒绝记录可能携带 incumbent 的性能，需要识别 measurement subject |
| `framework_baseline.json` 和 V1 crash progress | 已建立的框架基线、异常退出后的恢复信息 | V1 的 `progress.json` 是退出快照，不是持续更新的进度 |
| `trace-retention-manifest.json` | 最后一次终态记录、平台和硬件标识 | 旧终态文件可能保留到下一次运行结束，不能覆盖新的活动记录 |

上述目录均相对于单个 campaign 工作空间。读取器应兼容初始化阶段只有部分文件的情况。

## 插件结构

```mermaid
flowchart LR
    OP[优化器及框架子进程] --> WS[已有工作空间记录]
    LP[dashboard launch profile] --> CORE[Core 组装和 Bootstrap 调用]
    CORE --> UI[aka.dashboard 插件]
    CORE --> SOURCE[long-horizon 适配插件]
    WS --> SOURCE
    SOURCE --> CONTRACT[observation_source / 观测协议 v1]
    CONTRACT --> UI
    UI --> HTTP[本地只读 HTTP 接口]
    HTTP --> PAGE[浏览器进度页面]
```

优化器负责控制、验证和晋升。适配插件负责读取、汇总与语义转换，展示插件负责呈现；两者不参与 token 预算执行、canonical memory 写入、环境恢复或候选接受判断。读取器不调用会创建目录或修改 Git excludes 的 `CampaignStore` 构造函数。

`atrex-aka-dashboard` 依赖 Bootstrap；`atrex-aka-observation-long-horizon` 只依赖 Contracts，两者都不依赖优化器包。Dashboard 的 `long-horizon` 安装 extra 包含默认适配器。契约位于 `aka.contracts.observation`，Core token 由 Dashboard 的 `aka.tokens` entry point 注册；源码启动器显式提供同一 token。包归属相互独立：

```text
src/aka/dashboard/
├── __main__.py                # 源码启动器，仍通过 Bootstrap profile
├── plugin.py                  # Core 声明、构造和 cleanup 注册
├── startup.py                 # 同步 Startup 入口和观察目录参数
├── server.py                  # 显式路由和本地 HTTP 服务
├── tokens.py                  # observation_source 的 Core token
├── static/index.html          # 原生页面及样式、脚本
└── profiles/
    ├── dashboard.json         # Bootstrap launch manifest
    └── compositions/dashboard.json
src/aka/observation/long_horizon/
├── plugin.py                  # 提供 observation_source
├── reader.py                  # long-horizon 记录的只读投影和版本校验
└── demo.py                    # 仅向预览临时目录写入示例记录
```

`tools/build_distributions.py` 的 `OWNERS` 分别收集两个包的源文件。Dashboard 的 package data 包含页面、本地字体和 profile JSON；构建器先生成 sdist，再从 sdist 构建 wheel。

Dashboard 声明 `inject = ("observation_source",)` 和 `provide = ("startup",)`。`apply` 借用数据源构造 Startup 并注册 cleanup；数据源声明 `provide = ("observation_source",)`。HTTP 绑定和目录扫描在调用 Startup 后开始，导入或 boot 不启动服务。`close()` 幂等释放 socket 和请求线程；`run()` 的退出清理和 Core dispose 可以安全重复调用。

独立 profile 的目标为 `startup`，默认组合装载 Dashboard 和 long-horizon 数据源。它与优化 profile 使用不同 Root；跨进程以已有持久化文件为边界，不依赖同一 Root 的事件总线。数据源缺失时 boot 明确失败，不隐式创建默认读取器。

## 统一观测契约

`ObservationSource.protocol_version` 必须为整数 `1`。`open(workspace)` 创建线程安全的只读 `ObservationReader`，提供 `campaigns()`、`campaign(id)`、`episode(id, number)`。`create_preview(temporary_directory)` 仅向展示进程拥有的临时目录生成预览。实现不保存 Core Context/Root，Dashboard 不读取数据源的业务文件。

三个查询响应都包含 `observation_version: 1`；HTTP 和浏览器分别校验。未知数据源协议在 boot 失败，未知响应版本返回 HTTP 409，页面显示明确错误并保留上次数据。campaign ID 是不含路径分隔符的 URL-safe 字符串，不要求使用 long-horizon 的 hash。

| 契约字段 | 所有权与用途 |
| --- | --- |
| `display_name`、`empty_message` | 数据源提供用户可读名称和空状态文案；UI 不解析目录命名规则 |
| `stages: [{id, label}]` | 数据源提供有序流程阶段，页面动态生成流程标记 |
| `progress.stage_id`、`progress.phase_label` | 当前阶段标记与观测状态文案，保留 `phase` 作为源状态 ID |
| `phases: [{phase, label, usage}]` | 数据源提供有序 token 归因行，新增阶段不被 UI 固定列表丢弃 |
| `canonical.incumbent` | 携带数值、单位、指标名称及来源；数据源定义可比较的历史段 |
| `tokens` | 已结算总量、统计范围、精确/部分/不可用状态；阶段值不再次加入总量 |
| `issues` | 原生记录不兼容或更新中的诊断，不把不兼容格式当成等待或精确统计 |

默认适配器兼容未声明版本的历史记录。已声明的 live、telemetry、journal、framework baseline、crash snapshot 和 trace manifest 必须匹配各自已支持版本；原本未版本化的 state、attempt 和 canonical memory 不接受未知版本声明。新版本返回 `unsupported_schema_version`，不使用旧 schema 解读，受影响用量按已知控制记录降为 partial 或 unavailable。畸形写入仍沿用上次有效记录；版本不兼容会明确报告。

Long-horizon 适配器保留现有四个流程分组和九行 token 明细；发现额外 token 阶段时加入输出并提供名称，未知活动阶段也加入流程标记。其他 workflow 可提供完全不同的阶段列表，UI 无需修改。默认适配器仍只承诺 long-horizon 文件协议，新目录布局由新的适配插件负责。

替换数据源可向当前 composition 追加以下 patch，保留 Dashboard 配置：

```json
{
  "api_version": 1,
  "patches": [
    {"id": "observation-source", "remove": true},
    {"id": "my-source", "insert": {"name": "my_workflow.observation_plugin", "required": true}}
  ]
}
```

新模块提供 `observation_source`，实现上述契约。插件 `name` 遵循 Core 的模块命名规则，数据源依赖与身份文件由自身声明。安装包模式使用 `aka run --patch` 加载替换 patch；源码调用 `run_profile` 时通过 `tokens=(OBSERVATION_SOURCE,)` 显式传入 `aka.dashboard.tokens.OBSERVATION_SOURCE`。

## 启用方式

Launch manifest `profiles/dashboard.json`：

```json
{
  "api_version": 1,
  "composition": "dashboard",
  "target": "startup",
  "tokens": ["observation_source"],
  "required_services": ["observation_source"]
}
```

Sibling composition `profiles/compositions/dashboard.json`：

```json
{
  "api_version": 1,
  "patches": [
    {
      "id": "observation-source",
      "insert": {"name": "aka.observation.long_horizon.plugin", "required": true}
    },
    {
      "id": "dashboard",
      "insert": {
        "name": "aka.dashboard.plugin",
        "required": true,
        "config": {
          "host": "127.0.0.1",
          "port": 8765,
          "refresh_ms": 2000
        }
      }
    }
  ]
}
```

`Config` 只允许 `127.0.0.1` 监听，校验 0 至 65535 的端口及正整数刷新间隔。`port=0` 由系统分配空闲端口；指定端口占用时报告明确错误。静态资源纳入包数据和插件 `identity_files`，保持 Bootstrap 对实现及资源的身份验证。第一版沿用 strict resume policy。

源码目录中无需安装包即可启动：

```bash
PYTHONPATH=src python -m aka.dashboard --workspace /path/to/runs
# 让系统选择空闲端口：
PYTHONPATH=src python -m aka.dashboard --workspace /path/to/runs --port 0
```

页面使用英文界面。只预览有任务时的展示，可运行 `PYTHONPATH=src python -m aka.dashboard --demo --port 0`。演示包含探索中的 CUDA campaign、已接受与拒绝的轮次、阶段 token 分布以及用量不可用的 Triton campaign。页面显式标记 “Demo mode · Sample campaign data”；演示使用临时工作空间，不启动 Agent 或 GPU 作业，服务退出后删除示例数据。`--demo` 与 `--workspace` 互斥。

安装包的使用方式：

```bash
# 启动优化，继续使用现有 application profile。
aka optimize --repo-root /path/to/aka -- \
  --op-dir /path/to/problem --platform H20 --framework Cuda \
  --workspace /path/to/runs

# 另一个终端启动展示插件。profile 的绝对路径来自安装包或源码目录。
aka run --profile /path/to/aka/src/aka/dashboard/profiles/dashboard.json -- \
  --workspace /path/to/runs
```

可用 `python -c 'from pathlib import Path; import aka.dashboard; print(Path(aka.dashboard.__file__).parent / "profiles/dashboard.json")'` 获取已安装 profile 的路径。`--port` 参数放在 `--` 后，覆盖当前展示服务的监听端口。修改默认刷新间隔可使用 Core 的 `--patch`，配置替换时保留所需的 host、port 和 refresh_ms。

构建和安装 UI 及其依赖：

```bash
python -m pip install build
python tools/build_distributions.py --output /tmp/aka-dashboard-dist \
  atrex-aka-core atrex-aka-contracts atrex-aka-bootstrap atrex-aka-dashboard \
  atrex-aka-observation-long-horizon
python -m pip install /tmp/aka-dashboard-dist/*.whl
```

使用已发布的包时，`pip install 'atrex-aka-dashboard[long-horizon]'` 安装默认数据源；使用自定义数据源只需安装 Dashboard 基础包及对应适配包。

`--workspace` 接受单个 campaign，或包含 `kernel_opt_*` 的父目录。父目录只扫描直接子目录，不递归扫描 episode 私有工作树。安装包用户应使用包内 profile 的实际绝对路径；沿用现有 `aka run --profile`，第一版无需贡献新的 CLI 命令。

服务启动时打印实际浏览器地址，按 Ctrl+C 结束展示服务。优化结束后页面仍可查看历史记录。

## 页面内容

页面保持单屏主视图，桌面端把趋势和 token 明细并排，窄屏顺序排列。框架或 campaign 选择器只在有多个任务时出现。

| 区域 | 展示内容 | 交互 |
| --- | --- | --- |
| 任务头部 | 算子、框架、平台、最近一次观测阶段、数据更新时间 | 切换 campaign |
| 当前活动 | 当前 episode 和版本、fast/full/goal、实验数量、最新实验摘要 | 展开最近实验 |
| 结果概览 | canonical 最新版本、接受数、当前 incumbent 性能、已结算 tokens | 数据不完整时显示具体状态 |
| 性能趋势 | canonical incumbent 性能随版本变化 | 选择版本查看结果；拒绝轮保留原 incumbent |
| Token 用量 | 每个 episode 用量、累计已结算用量、所选 episode 阶段分布 | 选择 episode；展开输入、输出、缓存和原因码 |
| 轮次记录 | episode、版本、模式、接受/拒绝/pivot/blocked/interrupted、token、验证结论 | 查看同一记录的摘要、归因和证据文件名 |

默认适配器使用“准备 → 探索 → 验证 → 记录”标记当前阶段，其他数据源按自己的 `stages` 列表展示。探索可以包含多次实验，不能用阶段序号伪造完成百分比。fast 模式明确显示 evaluator 验证，full 模式明确显示 ABBA 验证。

存在正式配置时，额外展示“已记录版本 / 版本上限”和“预算已结算用量 / token 上限”。`--max-iters` 对应 canonical 版本上限，不等于 episode 数；`--token-budget=0` 表示无限额。第一版现有文件不完整保存这些配置，缺失时显示“预算未记录”，隐藏百分比。每个框架有独立预算，汇总页保留各任务的预算归属。

第一版将活动状态标为“最近观测：探索中”等，区分“页面连接正常”和“优化进程仍在运行”。没有新 journal 不代表失败；没有 active 文件也不代表完成。现有文件没有统一 run identity，所以终态显示为带时间的历史记录，不确认当前进程存活或当前运行已结束。旧 manifest 与新 active/live 记录矛盾时标记旧终态，保留各自时间，避免旧终态覆盖新运行。

## 数据和性能口径

读取器把文件投影成稳定的浏览器模型。现有 `memory/live.json` 只携带最近实验；轮次详情优先读取归档，活动 journal 只通过已登记的工作树位置解析。无法读取或工作树已移除时，展示 live 摘要和“详情暂不可用”。

当前 incumbent 曲线以已建立的 V0、框架基线和 `attempt.accepted=true` 的晋升记录为依据，并关联对应 canonical memory。接受记录需要正确性通过、完整测量和一致的版本绑定。拒绝轮的 candidate 测量仅出现在轮次详情中；携带旧 incumbent 性能的记录不产生新的性能提升。

优先使用记录的 `performance_objective` 和 `performance_score`；当目标是 `shape_speedup_arithmetic_mean` 时，曲线展示对应平均加速比。其他支持的记录可以展示 `latency_us_geomean`，标注微秒及基线。不同目标、硬件或 workload 的记录不连成可比较的曲线；缺失或不完整数据留空。

多任务页面只展示 task 级汇总，不读取隐藏 shape、私有 evaluator 输入或完整 Agent 对话。

## Token 口径

首页主值命名为“Episode Agent 已结算 tokens”，直接使用 `state.tokens`。它与现有预算口径一致，不能加上 `attempt.tokens` 或 telemetry terminal total 再求和。这些文件是同一次消耗的不同表示。

`state.usage_receipts` 用于解释已有去重结算，不由展示插件重写。当前调用尚未结束时显示“当前调用用量待结算”；页面仍然刷新实验进度。重启或重复读取同一文件不会新增消耗。

单个 episode 的详细用量优先使用 `telemetry.summary.json`，预算用量使用 `control_tokens` 或归档的 `attempt.tokens`。结构化 total 与 control total 不一致时分别标注，保留 `control_token_total_mismatch`，不能强行对齐。

| 观测状态 | 页面行为 |
| --- | --- |
| `exact` | 按记录展示数值和精确标记 |
| `partial` | 展示已观测数值和“部分统计”，允许查看原因 |
| `unavailable` 或 `null` | 显示“不可用”，图表留空 |
| Qoder 后端所有 usage 为 0 且适配器判定不可用 | 即使控制计数是 0，也显示“用量不可用” |
| 同会话恢复语义未确认 | 保留 `same_session_resume_usage_semantics_unqualified`；不用 invocation 累加值冒充精确总量 |

类型明细直接展示标准化的 input/output/cache read/cache write。总量采用 `TokenUsage.total_tokens`，避免在某些后端把已包含的缓存重复加上。默认适配器保留已有七阶段：profile、research、planning、implementation、correctness、benchmark、recording，另列 orchestration 与 unattributed，并保留记录中的其他阶段。阶段缺失显示不可用，不能平均分摊；归因质量与 terminal total 质量分别展示。

首页明确说明统计范围。setup、V1、独立 policy/numerical reviewer 和 `gen-plan` 辅助 CLI 消耗没有统一进入 `state.tokens`；第一版不把它叫作“全流程总 tokens”。多个任务汇总时，只要有未知用量就标为“已观测小计，部分任务用量不可用”。人民币、美元或计费额度需要明确的模型、费率和计费口径，不从 tokens 推算。

## HTTP 和读取约定

使用 `ThreadingHTTPServer` 与显式 `BaseHTTPRequestHandler` 路由，仅监听本机。这个服务定位为本地观察器。[Python HTTP 服务文档](https://docs.python.org/3/library/http.server.html)说明了这些接口的能力与适用边界。

| 接口 | 返回内容 |
| --- | --- |
| `GET /` | 包内静态页面 |
| `GET /api/campaigns` | 已注册观察目录中的 campaign 列表和轻量摘要 |
| `GET /api/campaigns/{id}` | 所选任务进度、性能历史、用量与数据质量 |
| `GET /api/campaigns/{id}/episodes/{episode}` | 所选 episode 的摘要和 telemetry 明细 |

浏览器 ID 映射到启动时允许观察的目录，不接受 HTTP 参数中的任意文件路径。未知 ID 返回 404，文本用 DOM `textContent` 呈现。接口不提供写入、执行、恢复或停止优化的动作。

首次读取历史记录，后续按文件大小和更新时间更新缓存；活动文件每轮刷新，完成的 episode 明细按需读取。多文件不构成原子事务，读取器校验 episode/版本关联；过渡期间保留上次有效值并标记“记录更新中”。初始化缺失显示等待，解析错误显示数据不可用，不能转成数值 0 或成功状态。只读操作不添加锁或轮询写入。

页面收到上一次响应后再安排下一次刷新，避免请求重叠。后台标签页降低刷新频率，重新可见时立即刷新，可使用 [Page Visibility API](https://developer.mozilla.org/en-US/docs/Web/API/Page_Visibility_API)。请求失败时保留数据并显示连接状态，不修改优化状态。页面和静态资源均可离线加载，不依赖 CDN。

## 实施步骤

| 步骤 | 范围 | 可验收结果 |
| --- | --- | --- |
| 第一版（已实现） | dashboard Core 插件、独立可安装包、profile、只读 reader/API、进度和用量页面 | 从现有工作空间直接观察多个 campaign；清楚区分已结算、不可用和待结算 |
| 配置和生命周期观测 | 在优化侧补中立的 campaign 元信息，含 run identity、真实预算、开始与结束时间、停止原因 | 页面可以展示预算和当前运行的明确终态；旧文件不覆盖新运行 |
| 流式用量 | Agent Runtime 提供标准化事件回调及累计 receipt，支持可靠的后端实时观测 | 运行中的用量按后端事件刷新；恢复不重复计数，结算后与 terminal usage 对账 |
| 全流程范围 | 给 setup、V1、独立评审和辅助 CLI 注册独立 invocation receipt | 已观测全流程总量可分角色查看，并与现有 episode 预算分开 |

实时用量需要补真实生产者：当前 `run_bounded()` 使用 `communicate()`，标准化事件主要在调用结束后处理，Codex rollout 也在返回后对账。刷新浏览器不能让这些数据自动变成实时数据。流式改动应保留当前进程取消、依赖防护和环境恢复行为，不能仅为 UI 改写预算算法。

后续生产者使用中立的 observation 契约，由 application 组装方显式注入；业务实现不保存或解析 `Context`、`Root`、`BootReport`。跨框架进程通过工作空间中的观测快照传递，不假设 Core 事件总线能跨 Root 或跨进程。UI 插件依旧只消费数据。

## 验收要求

自动检查：`PYTHONPATH=src:. python -m unittest discover -s packages/atrex-aka-dashboard -v`。fixture 使用现有 journal 和 telemetry 生产者核验真实 schema；HTTP 集成检查确认读取不会改写优化工作空间。安装包还需检查静态页面和 profile 资源，以及安装后的 `aka run` 启动。

- 默认优化 profile 不加载 dashboard；缺少 UI 包不影响优化命令。显式选择 dashboard 且插件缺失时，按 Core 必需项规则失败。
- 导入和 boot 不扫描目录或绑定端口；启动和 dispose 后服务资源正确释放。单任务与多框架任务都能显示，所有优化子进程不会额外启动 UI。
- 使用 fixture 核验 accepted/rejected/pivot/blocked/interrupted、fast/full/goal、V0/V1 与初始化阶段；canonical incumbent 不被未晋升候选或携带性能的拒绝记录改变。
- 核验 receipt 重放、同会话恢复、invocation 待结算、预算与观测 total 不一致、Qoder 不可用、阶段归因缺失及多任务部分统计。
- 优化器重启、工作树移除、旧 manifest 残留和文件更新过程中，页面保留可解释状态；UI 开关不改变 canonical memory、预算与接受结果。
- 浏览器检查 campaign 切换、episode 详情、连接失败和窄屏布局；页面刷新后仍能读到同一累计用量，离线不请求外部资源。

## 代码依据

- [Core 事件和生命周期](../src/aka/core/context.py)、[插件声明](../src/aka/core/declaration.py)以及[Bootstrap 调用与清理](../src/aka/bootstrap/host.py)。
- [默认 application 插件](../src/aka/legacy/application/plugin.py)和[中立 Startup 契约](../src/aka/contracts/startup.py)。
- [进度镜像](../long_horizon/journal.py)、[去重结算](../long_horizon/store.py)、[episode 会话结算](../long_horizon/session.py)和[终态归档](../long_horizon/campaign.py)。
- [阶段归因](../orchestrator/telemetry/phase_tokens.py)、[标准化用量模型](../orchestrator/agent_runtime/model.py)、[后端适配](../orchestrator/agent_runtime/adapter.py)和[V1 退出快照](../orchestrator/framework_baseline_progress.py)。
