# 容量控制权与外部保底

网关容量控制采用一个 profile 持有控制权、同 profile 多实例竞争共享租约的模型。正式和预发使用同一份 Hologres operations 事件表、租约表及相同集群配置名称；仅指向同一个 Ray 地址不足以协调。

## 集群登记与首次初始化

各网关仅使用自己的集群登记列表。共享控制键继续使用配置 `name`；运维须保证连接同一 dashboard URL 的正式、预发配置同名，独立集群使用不同名字。

启用的本地登记自动初始化 profile 归属，正常设置保底无需点击接管。初始化在集群租约内读取新鲜后端约束，保存完整外部 bundle、数量与 selector；已有网关目标不会变成永久外部保底。后端不可读、session 未知或约束无法完整保留时，不取得归属、不覆盖约束。已有其他环境归属时拒绝启用，须显式转交或先删除旧环境登记。

归属统一约束 Rust 路由、上传预选、普通作业准入、换池和阶段 GPU 准入，以及 SDK 提交桥和全部容量写入口。同环境实例共用持久归属、GPU 预约和集群操作租约。归属快照每 5 秒刷新，有效期 15 秒；读取失败不延长原授权，没有有效授权的集群关闭新调度，存量作业继续轮询、取消和清理。未配置共享账本的单环境部署按本地配置运行，不提供跨环境互斥。

外部保底独立于可到期的手动保底，采用 Ray 逻辑 GPU 单位。相同 bundle 与 selector 取各来源数量的较大值；不兼容的形状保守保留，CPU 等附带资源也保留。无法无损重放的原生亲和性或 fallback 约束关闭容量写入。清除手动保底不会清除外部保底。

## 正式与预发转交及删除

同一 `name` 任一时刻只属于一个环境，各环境的流量需求不聚合。转交请求先登记排空状态，旧环境停止新路由和普通作业准入，已有执行及其阶段继续收尾。等待旧归属快照和有界在途准入失效后，旧环境从共享运行态提供新鲜完整报告；活动执行、清理和隔离全部为零，才在集群租约内完成归属切换。SDK 提交前复核当前归属；它已在 Rust 中持久登记，纳入排空报告，不争用容量写入租约。排空安全窗口默认至少 90 秒，覆盖旧准入快照和有界 SDK 调用；后端约束和外部保底保留。

独立命令可以请求转交，但旧环境未升级、报告不可读或执行未排空时不会强抢。目标环境必须有自己的登记，转交不会创建目标环境配置。临时借用结束后按相同流程归还。排空尚未完成时，可在页面取消转交，或执行独立命令 `cancel-transfer`；删除排空期间重新发布 `enabled=true` 会取消该次删除排空。

`manage_clusters delete` 默认先发布停用登记，再等待存量执行和清理排空；共享协调器确认排空后释放归属。超时保持停用状态，排查后重试。`delete --force` 跳过等待，立即返回停用结果；历史绑定仍供已运行作业收尾。两种删除都不销毁 Ray、不主动取消作业，也不直接清空后端资源约束。

接管／转交／暂停／恢复入口沿用 BUC 全局管理员权限和版本校验，独立命令要求共享账本凭据；正常初始化由登记自动完成，`claim` 仅保留旧管理接口与人工恢复兼容。容量暂停与集群归属独立，暂停扩缩容不改变 owner。

## 外部应急操作

应急流程是「暂停控制成功 → 运维直接修改 Ray → 显式恢复」。暂停等待已取得集群租约的操作结束，保留后端当前约束，不依赖后端可达。暂停后重启网关仍不会写容量。

未先暂停而直接修改约束时，网关会在下一次实际写入或稳态复核中读取完整约束；同 session 的未知漂移使控制转为暂停，不自动修复。正常后端 session 重启且约束为空时，可以重放已登记保底。外部修改成当前目标完全相同的值无法区分写入方。

恢复默认保留原外部保底；仅在明确选择「将当前外部约束登记为新的外部保底」时替换它。恢复将重新读取后端约束并校验控制版本。

Ray 的请求接口没有外部 fencing／CAS。直接写 Ray 可能与在途网关请求竞争，因此自动检测漂移不能替代先暂停的流程。新旧网关共享同一账本和集群名时，由下述兼容租约协调滚动升级；不经过该租约的外部脚本仍须按暂停流程操作。

当前服务端将 Ray SDK 限定为 `>=2.56.1,<2.57`，锁定依赖和实机集群版本为 `2.56.1`。原生约束读取在目标 Ray 集群内执行，依赖目标版本的 GCS 生成协议；升级目标 Ray 时也须运行真实 SDK 的 selector 往返验证。符号缺失、协议读取失败或无法完整重放的约束均关闭容量写入，不以空约束替代失败。

在已暂停控制且原始约束为空的测试集群运行：

```bash
python scripts/verify_ray_capacity_selectors.py --config-file operator.json --cluster gpu-pool
```

该诊断作业不占 GPU，用已在线节点的 0.125 CPU 需求核对单值／多值 IN 与 NOT_IN 的原生 protobuf operator 和 values，并在结束前清空临时约束。原始约束非空时拒绝测试；session 变化或恢复失败须人工核对。它不创建真实 GPU 工作负载，不代替扩容与准入验收。

## 新旧版本滚动升级与回滚

兼容范围是使用同一份 operations 事件表、租约表且以同一 `name` 取得容量租约的旧网关。新版本先等待旧操作释放租约，再占用旧版使用的集群名租约；各新版实例使用独立操作租约串行写入。旧网关业务作业不因此停止，但后续容量操作拿不到旧入口。正式与预发仍按持久归属各管自己的集群，兼容租约不替代登记唯一性和排空转交。

首次交接、进程崩溃或写入结果未知时，保持旧入口保护，至少等待九十秒并读取完整 Ray 作业列表，确认 `agate-scale-` 控制作业全部结束后才允许新容量写入。后端不可读或控制作业未结束时继续等待，不以超时当作成功。旧账本中的写入仍为 pending 时，空作业列表不能单独证明安全。仅当完整回执历史证明后续失败均在调用后端前被归属校验拒绝，并有当前 Ray session 的成功回执、原生约束与原目标完全一致时，自动补记核对回执后交接；不改原保底及到期时间。超时、空错误或其他结果不明的失败继续保留保护，等待明确完成证据。新版控制作业另带执行期限，排队或初始化过久后拒绝写入。独立暂停、转交命令只更新账本，未核实的远端写入保留保护，后台继续核对；暂停成功仍需在运维页确认交接完成，再直接修改 Ray。

正常空闲时，兼容租约有效期默认九十秒，持有该集群的新版实例每五秒续租；任一新版实例存活即可延续保护。最后一个新版退出后，租约安全到期，旧版可自动恢复容量控制，无须人工删锁。回滚不清空原生约束，也不修改历史手动保底及到期时间。

以下情况不会自动解除保护：远端写入结果未知、控制暂停或排空、独立外部保底、包含多卡 bundle、CPU 等附带资源或 selector 的约束。旧版无法保持这些语义，直接恢复会丢失状态；运维页显示等待核对或回滚不兼容，应先恢复新版完成核对、收尾或显式调整控制状态。恢复后重新判断是否可以自动到期，不提供强制清锁快捷入口。

运维快照的 `capacity_control.compatibility` 提供 `phase`、`reason` 和 `lease_until`。`ready` 表示保护已生效且允许安全自动回退；`preparing`／`uncertain` 表示交接或核对未完成；`rollback_blocked` 表示当前状态不兼容旧版。该字段只描述容量控制，不能据此认定业务准入已就绪。

## 独立管理命令

在装有服务端包的运维环境运行以下命令。配置文件须明确包含当前 profile 和共用 Hologres 控制表配置及凭据；文件不得提交仓库。命令不调用网关 HTTP。

```bash
python -m app.manage_capacity status --config-file operator.json --cluster gpu-pool
python -m app.manage_capacity pause --config-file operator.json --cluster gpu-pool --revision CONTROL_REVISION --actor EMP_ID
python -m app.manage_capacity transfer --config-file operator.json --cluster gpu-pool --revision CONTROL_REVISION --target-profile pre --actor EMP_ID
python -m app.manage_capacity resume --config-file operator.json --cluster gpu-pool --revision CONTROL_REVISION --actor EMP_ID
```

从 `status` 的响应取得 `revision`。正常登记自动初始化归属；控制变更须携带当前版本。转交命令返回排空状态后，通过 `status` 查看完成进度；转交不会自动恢复容量暂停状态。
`status`、`pause`、`transfer` 不连接运行态 PostgreSQL、不解析资源目录、不联系后端。`claim`、`resume` 才解析目标集群并读取后端。命令只探测既有控制表，不自动建表。

应急修改后的当前约束需要成为新保底时，显式使用：

```bash
python -m app.manage_capacity resume --config-file operator.json --cluster gpu-pool --revision CONTROL_REVISION --adopt-external-floor --actor EMP_ID
```

管理员页面使用新增的 `PUT /dashboard/api/ops/ray/{cluster_name}/capacity-control`，请求包含 `action`、`revision`，转交附带 `target_profile`，恢复可带 `adopt_external_floor`。该接口沿用 BUC 全局管理员鉴权及同源意图头，不是 AK 机器接口。

## 0.14.17 升级预检与历史目标迁移

迁移期间冻结涉及的资源目录修改；新版本自动隔离旧版本的容量写入，无须人工停写旧网关。正式与预发不能同时启用同名集群；新实例在运行态就绪后接流，不等待所有集群登记成功，`healthz` 仅表示进程存活。单池异常只阻断该池，运维页展示登记状态。账本读取短暂失败时保留尚未过期的原授权，不延长有效期；未获授权或授权过期的集群停止新增调度，容量写入仍须实时确认归属并取得共享租约。

检查双方完整配置，`--peer-config-file` 可重复指定，必须包含所有使用该共享账本的其他环境。命令解析各自当前 PostgreSQL 目录；只读预检不改登记或后端约束，存在阻断项时退出码为一。

```bash
python -m app.manage_capacity preflight --config-file operator.json --peer-config-file peer.json
```

先处理重复登记：从旧环境停止新准入并排空后显式转交，停用旧环境登记。已经存在新协议 owner 的集群不能通过迁移命令抢占。

只有历史目标而没有新登记时，使用预检返回的 `revision` 和 `catalog_digest`：

```bash
python -m app.manage_capacity migrate-legacy --config-file operator.json --peer-config-file peer.json \
  --cluster gpu-pool --revision LEGACY_REVISION --catalog-digest CATALOG_DIGEST \
  --actor EMP_ID
```

命令先通过兼容租约隔离旧写入，等待远端控制作业结束；交接尚未完成时返回 `capacity_handoff_pending`，稍后重试。`--confirm-legacy-writers-stopped` 仍接受但已废弃，不再要求人工确认。命令在共享操作租约内复核两边登记与版本、读取完整原生约束，仅在约束与已知历史目标一致时写入归属；不调用容量写接口，不改原手动保底及到期时间，也不把它变成永久外部保底。约束漂移、版本变化、重复登记或后端不可读都会拒绝迁移。迁移后重新运行预检，再验证实际提交、运行和收尾。

登记刷新、关键控制操作与观测各有独立连接预算，运维快照顶层的 `control_pools` 提供占用、等待及借连接超时计数；原集群条目中的同名字段保留兼容；登记授权仍每五秒刷新、十五秒过期，不因本地繁忙延长授权。


### 逐波发布与集群故障隔离

首次上线或 owner 重新就绪时，VIP 就绪仅依赖网关运行态，不以全部集群登记成功为前提。未登记、归属冲突、排空或授权过期的集群继续停止新路由和准入；已有有效授权的集群正常服务。即使暂时没有可用集群，网关也可上线提供查询和运维能力，对没有可用后端的提交返回原有不可用响应。登记摘要 `ready` 仍表示所有启用集群均已获准接收新作业，是运维汇总，不是 VIP 上线条件。启动依赖初始化或 Rust 运行态恢复失败仍会阻止实例就绪。

1. 每波开始前，对当前正式和预发目录重新执行上面的 `preflight`，确认没有重复登记，查看各集群兼容交接状态。该命令只检查登记与历史目标，不探测后端，不能代替初始化验收；单个集群的待迁移、归属冲突或后端异常可隔离后继续发布，不能据此放开该集群的调度或容量写入。
2. 未就绪集群自动停止新路由和准入，可在发布后单独运维。需要显式排空或避免继续尝试初始化时，记录当前目录版本和该集群配置，仅将其 `enabled` 改为 `false`。数据库目录可用下列命令，版本号须替换为本次 `show` 的结果；inline 来源在原配置中停用该集群。保留存量执行与清理；排空后可能释放归属，不销毁后端或取消作业。

   ```bash
   AGATE_CONFIG_FILE=/secure/gateway.json python -m app.manage_clusters show
   AGATE_CONFIG_FILE=/secure/gateway.json python -m app.manage_clusters update \
     --name example-pool --set 'enabled=false' --expected-version CURRENT_VERSION \
     --author OPERATOR --reason "temporarily disable unavailable cluster before rollout"
   ```

3. 先发布一波，确认 Rust 运行态及 VIP 就绪，并在本次保留服务的集群上验证实际作业可完成，再推进下一波。只有 `/healthz` 成功不足以判断请求可用；登记摘要 `ready=false` 不阻止上线，但须记录不可用集群及其业务影响。旧实例通过兼容租约自动让出容量写入，仍可处理业务作业；不手工创建上线标记。若所有实例已离线，仍可从可连接数据库的运维主机使用管理命令修复目录。
4. 后端恢复后，重新核对双方当前目录和归属，确认其他环境没有启用或持有该集群。读取最新版本，仅将该集群恢复为 `enabled=true`，等待重新登记并提交实际作业验证。使用相同的 `update` 命令与最新 CAS 版本，不回滚整份旧目录，以免覆盖期间其他集群的修改。

运维页区分“登记尚未完成”和“登记初始化失败”；后者应检查登记日志，数据库连接繁忙则等待重试并检查连接池指标。连接池繁忙继续使用原 `runtime_database_pool_busy` 标识与日志前缀，已有匹配规则无需因本次升级改名。
