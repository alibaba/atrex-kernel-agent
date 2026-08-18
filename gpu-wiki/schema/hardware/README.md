# Schema — `hardware_wiki`(硬件事实参考库)

与 `kernel_wiki`(优化经验库)**平级且独立**的一个库。存的是硬件与 ISA 的**事实**,不是优化经验。

## 为什么必须独立

不是「内容不同」,而是**检索问题的性质不同**:

| | 经验库 `kernel_wiki` | 本库 `hardware_wiki` |
|---|---|---|
| 知识性质 | 某人试过、结果如何——可证伪、有成功率 | 厂商与 ISA 定义——不可证伪 |
| 检索语义 | 模糊排序:症状 → 候选手法 top-k | 精确查询:给定地址 → 唯一答案 |
| 排序 | 需要 `worth`(重要性、档位、反馈) | **没有排序**,也没有 `worth` 这一层 |
| 零命中 | 返回标注过的随机兜底样本 | **必须报错**——问「BF16 峰值」时给随机样本比不回答更糟,因为这个数会被当 roofline 的分母 |
| 反馈闭环 | 必需,驱动排序 | 无意义;只有勘误与版本换代 |

所以本库的 `tools/query_hardware.py` 是 **lookup**,不是 ranked search:无排序、无兜底、地址不认识就 fail-loud。

## 边界:什么进本库,什么留经验库

**判据只有一条:这句话可以被我们的 benchmark 证伪吗?**

- 不能证伪 → 事实 → 本库。例:`tcgen05.ld.red` 的语法与修饰符词表;TMEM 每 SM 256 KB;B300 的 INT8 峰值 187.5 TOPS。
- 能证伪 → 经验 → 经验库。例:「LDGSTS 在 32 KiB in-flight 附近饱和」「CLC 在均衡 GEMM 上并不总是更快」「某手法保留率 26%」——这些是实测、依设备与负载而变。

来源页混合两类内容时**必须拆开**。例如 B200 tensor-core 分析页的 §12 微基准、CLC 页的 Performance Impact 小节,都明确写着「这些是 B200 观测值,不是架构保证」——它们留在经验库。

`check_hardware_wiki.py` 的 **no-advice** 门会拦下混进来的推荐语(「usually faster」「we recommend」「保留率 N%」)。

## 三种记录类型

| type | 一条记录 = | 地址 |
|---|---|---|
| `spec-sheet` | 一款芯片的全部数值 | `--product b300` |
| `instruction` | 一个 ISA 指令族 | `--instruction tcgen05.ld.red` |
| `arch-feature` | 一项架构能力 | `--feature fp4-k96-2cta` |

统一信封:`identity`(定位) + `facts`(内容,给 agent) + `provenance`(证据) + `status`。**没有 worth。**

## 证据分级是强制的

每个数字都要能回答「这是谁说的」。`provenance.evidence_class` 三档:

- `vendor-published` —— 厂商为这颗芯片印出来的值
- `derived-from-system-total` —— 由 n 卡整机数值推算(**必须写清除数**,由 provenance 门强制)
- `architecture-analysis` —— 第三方或推断,**视为暂定**,优先用运行时设备属性

一张厂商页里经常混着多档,所以 `memory` / `compute_units` 支持 `provenance_overrides` 做**字段级例外**。B300 就是实例:容量与带宽是厂商发布,而 L2 126 MB 属架构分析。

**未发布的字段一律留 `null`,并在 `facts.unavailable` 里写明如何获得。** 编一个峰值比缺一个峰值危险得多——它会静默污染所有由它算出的利用率。fabrication 门强制这条:`null` 必须有说明,有说明的字段不许同时带值。

## 用法

```bash
# 取 roofline 的分母,带证据等级
python3 tools/query_hardware.py --product b300 --field peak_compute.bf16.dense
# → {"value": 2250, "provenance": "architecture-analysis", ...}

# 未发布的字段:给处置办法,不给替代值
python3 tools/query_hardware.py --product b300 --field compute_units.shared_memory_kb_per_sm
# → {"value": null, "unavailable": "...Query the CUDA device attribute at runtime..."}

# 跨代对比:防「新一代一定更强」的想当然
python3 tools/query_hardware.py --product b300 --vs b200
# → int8 −95.8%、fp64 −96.6%、fp4 +50%

# 指令与特性
python3 tools/query_hardware.py --instruction tcgen05.ld.red
python3 tools/query_hardware.py --feature fp4-k96-2cta
python3 tools/query_hardware.py --capability sm_103 --list features
```

约定与经验库一致:**stdout 只有一个 JSON**,提示信息走 stderr,所以可以直接 `json.load`。

## 维护

```bash
python3 tools/build_hardware_index.py         # 重建 records/index.json
python3 tools/build_hardware_index.py --check # 检查索引是否与记录一致
python3 tools/check_hardware_wiki.py          # 六道门
python3 -m unittest discover -s tools         # 查询契约测试
```

六道门各自防一类**静默**损坏:`schema`(记录漂移)、`ids`(id 与路径不符,引用解析不到)、`index`(记录不可达)、`provenance`(数字没有出处)、`no-advice`(经验混进事实)、`fabrication`(编造未发布的数字)。

## 现状

30 条记录,全部由本仓库策展文档投影而来:

| type | 条数 |
|---|---:|
| `spec-sheet` | 9 |
| `arch-feature` | 10 |
| `instruction` | 11 |

覆盖的产品:b200、b300、mi300x、mi308x、mi355x、sm120。

### 产品名称映射

`tools/hardware_identity.py` 是所有查询入口共享的固定身份表。表中的“内部地址”
与 `hardware_wiki/records/index.json` 的 `product` 完全一致；大小写、厂商前缀、
空格、下划线和 `GPU` / `accelerator` 后缀只影响输入写法，不产生新的内部地址。

| 内部地址 | vendor | arch | 允许的纯格式变化示例 |
|---|---|---|---|
| `b200` | nvidia | blackwell | `B200`、`NVIDIA B-200 GPU` |
| `b300` | nvidia | blackwell-ultra | `B300`、`nvidia-b300` |
| `mi300x` | amd | cdna3 | `MI300X`、`AMD Instinct MI-300X GPU` |
| `mi308x` | amd | cdna3 | `MI308X`、`amd_mi308x` |
| `mi355x` | amd | cdna4 | `MI355X`、`AMD MI-355X accelerator` |
| `sm120` | nvidia | blackwell-geforce | `SM120`、`sm_120`、`SM-120` |

例如 `NVIDIA B300 GPU` → `b300`、`AMD Instinct MI308X GPU` → `mi308x`。
A100/A800/A30/A10、L20/L40S/L4、
H100/H200/H800/H20/GH200、B100/GB200/GB300、RTX PRO 5000/RTX 5090/5080、MI300A/MI350X
也在同一身份表中；它们目前没有产品 spec sheet，因此返回 `not-recorded` 获取办法，
不会借用其他产品的数字。

架构名不会被强制映射到某个产品：例如 `gfx942` 同时覆盖多个 CDNA3 SKU，不能
据此选择 `mi300x` 或 `mi308x`；`GB200` 也保持为独立产品，不借用 B200 spec sheet。
Query 侧只消除大小写、空格、`-`、`_` 和明确的厂商包装词，不允许把一个身份翻译
成另一个身份。任何非产品标识都会 `unknown-product`，不会被映射到某张 spec sheet。
运行 `python3 tools/query_hardware.py --list products` 可查看内部地址、格式规则和已识别
但尚无 spec sheet 的产品。

**证据分级说明**:这些页面是第三方策展文档,不是厂商数据表,所以每条记录的
`provenance.evidence_class` 一律为 `architecture-analysis`——按 schema 定义视为
**暂定值**,优先用运行时设备属性核对。本库不会替厂商背书:任何数字都不会被标成
`vendor-published`。
