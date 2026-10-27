# 运维与特权能力的边界

仅在自部署、容量管理、故障诊断或 raw dev 时读取本页。普通 GPU 作业沿用主文档的 CLI 流程。
容量控制的实际管理命令与交接步骤见 [capacity-control.md](capacity-control.md)。

## raw dev

`agate dev` 是特权 raw shell，默认关闭。调用 AK 必须位于 `auth.dev_allowed_aks`，
部署方须确认目标集群的网络与执行隔离；`supported_kinds` 未配置表示允许全部 kind，包括 dev。
已有 `check`／`eval`／`profile`／`disassemble`／`env` 能完成的工作优先用对应接口。

```bash
agate dev 'python run.py' --gpu H20 --file run.py=./run.py --intent custom_harness
```

raw dev 强制回收，`--no-recycle` 被拒绝；asicli 通过容器 PID 1 代际变化确认回收，
不具备同等隔离的后端不应启用。字符串拦截只是纵深防护，不能替代网络策略与 RunD／Kata 隔离。
命令正文不进入普通请求日志、审计或 Hologres，允许／拒绝记录身份、摘要、字节数和有界风险标签。
集团浏览器登录不替代 dev 的 AK／SK 和专用 ACL。

已确认终态、可重试的基础设施故障按 `max_attempts` 有限重试，默认首次加一次，命令需容忍重复执行。
提交期断连或慢提交确认原收据／原身份，最多确认 60 秒，结果不明不会重新执行。
`dev_submission_ambiguous` 须核对外部副作用后再决定是否新建作业。
显式空 ACL 回 `403 dev_disabled`，AK 不在名单回 `403 dev_acl_denied`，
所有匹配 cluster 禁用 dev 回 `400 dev_cluster_disabled`。
默认作业超时 300 秒，默认部署可调至 1800 秒，实际按命令策略限制，代码绝对上限为 10800 秒。

## 容量、队列与故障接管

扩容目标不等于已就绪 GPU，等待实际容量时继续轮询原作业。容量未知／陈旧时使用保守额度，
不能用节点规格替代在线供给，也不能因可达状态正常就认定容量已就绪。
默认每池未完成 GPU 单位预算约为在线容量的两轮，常规单 AK 约为一半，多卡按卡数计，零卡至少一个单位。
管理员可热更新预算；容量缓存 TTL 不决定排队时长。

未执行作业可在提交时匹配的兼容池间自动切换，硬件、GPU 数量、放置、kind 和 OSS 约束仍生效。
原池停用／移除且无兼容候选回 `infra/resource_pool_unavailable`；短暂容量故障或目标队列满保留原期限等待。
未知提交不重放，已执行 dev 不因换池而重新执行，新加入或执行配置变化的池不自动成为旧作业候选。
故障实例可由健康实例接管，原 `job_id`、幂等键与远端 attempt 保留。
只有管理员明确确认原集群销毁时，才以 `backend_cluster_destroyed` 收口，重新执行需新请求。

Ray 保底、跨环境归属和兼容租约按容量控制参考执行；直接修改 Ray 前先暂停网关容量写入。
网关上线不等待所有集群登记就绪，未就绪池单独停止新路由和准入。

## 自部署与观测

服务端使用 Rust coordinator，无 Python coordinator 回退；单实例 SQLite、多实例 PostgreSQL 走同一作业入口。
SQLite 升级须停止旧 owner、备份数据库并提供匹配 Rust binary，删除旧 `off/passive/active` 配置。
PG 按作业 claim／epoch 接管，跨库迁移须停旧写入者并离线校验，不双写运行态。
PG 滚动升级在旧成员退出后启用新运行态语义，回滚需支持同等协议。
运行态数据库暂不可用时暂停接单，恢复租约与持久状态后继续原作业；不能重放结果未知的写事务。
`coordinator_capability_unsupported` 须部署方核对能力配置，不能换幂等键重提。

本机 GPU 是在 GPU 节点以 `python -m app.main` 部署网关并配置 `local` 集群，客户端仍通过 HTTP 指向该服务。
local 以网关用户权限运行，不提供 Pod／容器隔离；客户端和服务端仍分别安装。

Holo 观测与 operations 独立于运行态。三类观测事实的默认交付仍由网关执行；显式
`observation_mirror.delivery_mode=external_cdc` 才交给外部 CDC，保持 PG 持久化并暂停源清理。
切换须完成外部链路、容量预算和安全清理验收；回退须处理历史积压，不能仅关闭镜像开关。
PG 写入、领取／确认、维护和 Holo 投递故障分别显示，维护失败不等于作业执行失败。

首页作业构成和耗时按全环境终态时间窗口统计，包含取消，不是提交量或当前在跑数。
可靠性失败率只计基础设施故障，超时另报；账本不可用或缓存陈旧时保留未知／陈旧标记。
`/ops` 展示授权范围内实例健康与异常运行态，采样中断不等于租约失效，不直接改写运行态。
单次 SQL 取消只影响原请求；`GET /dashboard/api/stats` 中
`business_runtime.http_runtime.workers[].state_store.query_cancellations` 是进程内累计数，
结合 worker 的 `age_s` 判断新鲜度，重启归零，SQLite 下不可用。

钉钉监控的频率、阈值、窗口、开关及发送配置支持 Diamond 热更新；调参保留事件，关闭类别清除该类事件，
恢复流量检测时重新预热基线。清理告警默认关闭，PG 共享投影不能把多实例读数相加。

## TaskModule preview

TaskModule 是自部署 preview，没有 `agate` 子命令；使用 `atrex-gateway module validate|test` 与 `/v1/modules`。
在线 preview 默认关闭，仍要求 trusted AK、cluster 的 `module`／`runtime_classes` opt-in。
SQLite registry 仅内存，PG registry 共享持久化；不是通用生产制品服务。
默认注册／读取只返回文件摘要，input/source 不落 request log。schema 是拒绝 `$ref`／regex／组合递归关键字的有界子集，
不支持 OSS 附件；终态 JSON 最多 256 KiB，更大内容用声明 artifact。
Ray stage admission 必须保持原生 Dashboard／JobDetails 仅 operator 可见，因为 callback bearer 会交给 driver；
不能满足该控制面边界时保持关闭。
