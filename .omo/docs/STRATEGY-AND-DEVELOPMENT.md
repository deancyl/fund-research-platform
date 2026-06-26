# 中国基金/指数基金 AI 投研平台 — 策略与开发文档

**日期**: 2026-06-26 | **研究时长**: 2+小时 | **来源**: 60+学术论文、40+开源仓库、12维度并行研究
**覆盖**: AI决策架构 · 策略库(20+策略含胜率) · 回测引擎 · 策略切换 · 投前验证 · 风险管理

---

## 目录

1. [系统总体架构](#1)
2. [AI 决策引擎设计](#2)
3. [策略库（含胜率数据）](#3)
4. [回测引擎架构](#4)
5. [策略切换与市场状态识别](#5)
6. [投前验证框架](#6)
7. [风险管理体系](#7)
8. [实施路线图](#8)

---

## <a name="1"></a>1. 系统总体架构

### 1.1 Headless Core 双端架构（审计后重构）

原方案为 TUI 单体架构。经外部审计，现重构为 **Headless Core（无头核心引擎）+ 双端适配器** 模式，确保一套后端逻辑同时支撑 TUI 终端和 Web 视图。

```
                         ┌────────────────────────────────────────────────┐
                         │        Headless Core (独立核心引擎)              │
                         │  所有模块仅使用标准 Python 类型 / Dict / DataFrame │
                         │  绝对不导入 textual / fastapi / 任何 UI 框架      │
                         ├────────────────────────────────────────────────┤
                         │  Layer 5: AI 决策引擎                            │
                         │  ├── LangGraph 多智能体辩论 (Macro/Quant/Risk/CIO)│
                         │  ├── 8因子评分 + 置信度校准                       │
                         │  └── DecisionHub (确定性决策中枢)                 │
                         ├────────────────────────────────────────────────┤
                         │  Layer 4: 策略层                                 │
                         │  ├── 20 策略库 (动量/均值回归/因子轮动/行业轮动)     │
                         │  ├── 策略切换引擎 (HMM 状态识别 → 动态映射)        │
                         │  └── TurnoverController + CrowdingMonitor       │
                         ├────────────────────────────────────────────────┤
                         │  Layer 3: 回测与验证层                            │
                         │  ├── 向量化回测 (VectorBT) + 事件驱动 (Backtrader) │
                         │  ├── 中国市场规则引擎 (15:00截点/赎回费/CashLock)  │ ★ NEW
                         │  ├── WFO + CPCV + PBO + DSR 三重过拟合检测        │
                         │  └── Monte Carlo 鲁棒性检验                       │
                         ├────────────────────────────────────────────────┤
                         │  Layer 2: 分析引擎层                              │
                         │  ├── MyTT 技术指标 (中国兼容 KDJ/MACD/BOLL/RSI)    │
                         │  ├── fincore 风险指标 (Sharpe/Sortino/Calmar/VaR) │
                         │  ├── jh-factors CH-3 中国因子模型                 │
                         │  ├── 绩效归因 (Brinson + 因子归因)                 │
                         │  └── 新闻情绪分析 (bardsai + SnowNLP + jieba)     │
                         ├────────────────────────────────────────────────┤
                         │  Layer 1: 数据层                                 │
                         │  ├── AKShare (主力) + 东方财富直连 (备用)          │
                         │  ├── Baostock (指数补充)                          │
                         │  └── SQLite + DuckDB + Parquet (本地缓存)         │
                         └──────────────┬─────────────────────────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
   ┌──────────────────────────┐               ┌──────────────────────────┐
   │  TUI 适配器 (本地零延迟)    │               │  Web 适配器 (富图表/异构)  │
   │  src/tui/                 │               │  src/web/                 │
   ├──────────────────────────┤               ├──────────────────────────┤
   │ • Textual App 框架        │               │ • FastAPI REST 服务       │
   │ • 键盘精灵 / F5-F9 快捷键 │               │ • Vue 3 / React 前端      │
   │ • asyncio Worker 绑定     │               │ • ECharts 可视化          │
   │ • 直接 import core 模块   │               │ • WebSocket 实时推送      │
   │ • 零网络开销              │               │ • JSON API 异步调用       │
   └──────────────────────────┘               └──────────────────────────┘
```

### 1.2 目录结构（Headless Core Monorepo）

```
fund-research-platform/
├── src/
│   ├── core/                        # ★ Headless Core — 零 UI 依赖
│   │   ├── __init__.py
│   │   ├── data/                    # Layer 1: 数据采集
│   │   │   ├── fetcher.py           #   AKShare + 东方财富 数据获取
│   │   │   ├── cache.py             #   SQLite/DuckDB 缓存层
│   │   │   └── schema.py            #   数据 Schema 定义
│   │   ├── analysis/                # Layer 2: 分析引擎
│   │   │   ├── indicators.py        #   MyTT 技术指标
│   │   │   ├── risk_metrics.py      #   fincore 风险指标
│   │   │   ├── factor_model.py      #   jh-factors CH-3
│   │   │   ├── attribution.py       #   Brinson 绩效归因
│   │   │   └── sentiment.py         #   新闻情感分析
│   │   ├── engine/                  # Layer 3: 回测引擎 ★ 中国市场规则
│   │   │   ├── backtest.py          #   向量化 + 事件驱动核心
│   │   │   ├── order_cutoff.py      #   15:00 申赎截点验证
│   │   │   ├── redemption_fee.py    #   2026 阶梯赎回费计算
│   │   │   ├── cash_lock.py         #   资金交收延迟状态机
│   │   │   ├── capacity_gate.py     #   大额申购/巨额赎回限制
│   │   │   ├── slippage.py          #   A 股 ETF 滑点模型
│   │   │   └── validation.py        #   CPCV + PBO + DSR
│   │   ├── strategy/                # Layer 4: 策略库
│   │   │   ├── base.py              #   策略基类
│   │   │   ├── momentum/            #   动量策略 (S1-S5)
│   │   │   ├── mean_reversion/      #   均值回归 (S6-S10)
│   │   │   ├── factor_rotation/     #   因子轮动 (S11-S14)
│   │   │   ├── china_specific/      #   中国特色因子 (S15-S17)
│   │   │   ├── portfolio/           #   组合策略 (S18-S20)
│   │   │   └── switching.py         #   策略切换引擎
│   │   ├── recommend/               # Layer 5: AI 决策
│   │   │   ├── agents/              #   多智能体 (Macro/Quant/Risk/CIO)
│   │   │   ├── debate.py            #   辩论协议
│   │   │   ├── decision_hub.py      #   确定性决策中枢
│   │   │   ├── calibration.py       #   置信度校准
│   │   │   └── auditor.py           #   幻觉审计
│   │   └── api.py                   # ★ 统一服务层 API 入口
│   │                                #    (供 TUI 和 Web 共同调用)
│   │
│   ├── tui/                         # TUI 适配器
│   │   ├── app.py                   #   Textual App
│   │   ├── panels/                  #   面板组件
│   │   ├── commands.py              #   键盘精灵
│   │   └── worker.py                #   asyncio Worker 绑定
│   │
│   ├── web/                         # Web 适配器
│   │   ├── server.py                #   FastAPI 服务
│   │   ├── routes/                  #   REST API 路由
│   │   └── ws.py                    #   WebSocket 推送
│   │
│   └── cli/                         # CLI 入口
│       └── main.py                  #   fund-research tui / web / update
│
├── tests/                           # 测试（与 src/ 同构）
├── config.yaml                      # 全局配置
├── pyproject.toml                   # 依赖管理
└── README.md
```

### 1.2 技术选型总结

| 层级 | 技术 | 选型理由 |
|------|------|----------|
| **语言** | Python 3.11+ | 所有核心库均为Python生态 |
| **TUI** | Textual (26K★) | 金融终端先例(FinTerm/StocksTUI)，CJK已修复 |
| **主力数据** | AKShare (19.8K★) | 全部免费，基金/ETF/指数全覆盖 |
| **技术指标** | MyTT + pandas-ta | 通达信兼容KDJ/MACD，TA-Lib不兼容中国市场 |
| **风险指标** | fincore + jquantstats | 150+指标，Sharpe/Sortino/Calmar/VaR |
| **因子模型** | jh-factors (CH-3) | 中国三因子，剔除壳污染 |
| **回测(快)** | VectorBT | 向量化，比事件驱动快20-100x |
| **回测(准)** | Backtrader | 事件驱动，支持T+1/涨跌停 |
| **情绪分析** | bardsai 96.1%准确率 + SnowNLP | 中文金融情绪专用 |
| **LLM** | DeepSeek API + Ollama本地 | 中文最优，离线可运行 |
| **存储** | SQLite + Parquet | 零配置、便携、列式压缩 |

---

## <a name="2"></a>2. AI 决策引擎设计

### 2.1 总体决策架构

基于对 openInvest、FinAgent、TradingAgents-astock 三个生产级系统的源码级分析，我们采用 **混合决策架构**：

```
阶段1: 定量分析                     阶段2: 定性辩论                    阶段3: 决策融合
┌──────────────────┐        ┌──────────────────────┐        ┌──────────────────┐
│ 8因子基金评分系统  │        │ 多智能体辩论委员会      │        │ 确定性决策中枢     │
│                   │        │                       │        │                   │
│ • 收益因子 (15%)  │   →    │ Macro Strategist      │   →    │ 因子权重: 60%     │
│ • 趋势因子 (10%)  │        │   ↓                   │        │ 辩论权重: 40%     │
│ • 风险调整(20%)   │        │ Quant Analyst         │        │                   │
│ • 最大回撤(15%)   │        │   ↓                   │        │ 置信度校准       │
│ • 基金经理(10%)   │        │ Risk Officer          │        │ (Isotonic Reg.)  │
│ • 机构评级(10%)   │        │   ↓                   │        │                   │
│ • 规模流动(10%)   │        │ CIO 综合征税           │        │ 最终输出:         │
│ • 动量因子(10%)   │        │                       │        │ BUY/ACCUMULATE/  │
└──────────────────┘        └──────────────────────┘        │ HOLD/TRIM/SELL   │
                                                            │ + 置信度 0-1     │
                                                            └──────────────────┘
```

**核心原则**: LLM负责分析推理 → 确定性数学负责最终决策 → 置信度校准控制仓位

### 2.2 多智能体辩论协议

参考 openInvest 和 FinAgent 的生产级实现：

```
Round 1 (平行):
  Macro Strategist: 分析宏观指标(VIX/利率/汇率) → SIGNAL + STRENGTH
  Quant Analyst:   分析技术指标 + 因子得分 → SIGNAL + KEY_DATA
  Risk Officer:    分析集中度/现金/风控 → SIGNAL + 限制条件

Round 2..N (交叉挑战):
  Quant Analyst 看到 Risk Officer 的输出 → 反驳或修正
  Risk Officer 看到 Quant Analyst 的输出 → 反驳或修正
  
收敛判断:
  SIGNAL一致 + STRENGTH差距 < 1.0 连续2轮 → 提前终止
  或达到最大轮次(max 5轮) → 强制终止

CIO 综合判决:
  输入: 全部辩论记录 + 8因子得分
  输出: VERDICT (BUY/ACCUMULATE/HOLD/TRIM/SELL)
        + CONFIDENCE (0.0-1.0)
        + EXECUTION_PLAN (lump-sum/pyramid/grid)
        + RISK_PLAN (stop_loss, worst_case_pnl)
```

### 2.3 信息隔离协议

每个 Agent 只能看到特定信息，防止群体思维：

| Agent | 可见信息 | 不可见信息 |
|-------|----------|------------|
| **Macro** | VIX/利率/汇率/PMI | 持仓、技术指标 |
| **Quant** | K线/RSI/MACD/KDJ/8因子得分 | 持仓、宏观信号 |
| **Risk** | 仓位集中度/现金/最大回撤/过往教训 | 技术指标、宏观信号 |
| **CIO** | 全部Agent输出 + 持仓上下文 | —（唯一全知角色） |

### 2.4 置信度校准（防过度自信）

两种机制确保AI推荐不过度自信：

**机制1: Isotonic Regression校准**
```python
# 使用PAV算法对CIO的CONFIDENCE进行非参数单调校准
# 需要累积 ≥50 条 (prediction, outcome) 对的样本
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
calibrated_confidence = calibrator.fit_transform(raw_confidences, outcomes)
```

**机制2: 确定性审核（6项安全检查）**
- 过度自信检测: CONFIDENCE > 0.9 且样本不足50条 → 降至0.7
- 仓位限制: 建议配置不超过总仓位的20%（单标的）
- Agent不可用检测: 某Agent失败 → CIO被告知并降权
- 集中度覆盖: 持仓 >30% CONCENTRATION_PCT → 禁止HOLD
- 独立防御标记: 市场危机信号 → BUY降级至ACCUMULATE
- 冲突升级: Quant与Risk完全相反 → CIO必须给出显式理由

### 2.5 记忆与学习系统

**三阶段记忆整合**（参考 openInvest Dreaming + FinAgent Reflection）:

```
Light Sleep (每日):
  记录当日VERDICT + 7日后真实回报 → 更新短期校准数据

REM Sleep (每周):
  聚类分析: 哪些市场条件下AI表现好/差？
  生成2-4句可操作教训

Deep Sleep (每月):
  因子权重重校准: 基于滚动12月绩效
  淘汰衰减因子，引入新因子
```

---

## <a name="3"></a>3. 策略库

### 3.1 策略分类总览

共 **20个策略**，分5大类。每个策略附带: 参数、适用市场状态、历史胜率、学术/实证来源。

---

### 3.1.1 动量与趋势类（5个策略）

#### S1: 因子动量轮动 ⭐⭐⭐⭐⭐ 最高收益

| 属性 | 内容 |
|------|------|
| **来源** | Ma, Liao & Jiang (2024) Journal of Empirical Finance |
| **逻辑** | 10个常见因子(价值/动量/质量/低波/规模等)每月排名，全仓买入排名最高的因子组合 |
| **回看期** | **1个月**（短期最优） |
| **年化收益** | **9.91%** |
| **夏普比率** | **1.15** |
| **最大回撤** | -12.4% |
| **胜率** | ~65%月度胜率 |
| **中国市场关键差异** | 中国因子动量夏普1.15 vs 美国0.61 — 散户噪音使错误定价持续更久 |

**实现要点**:
```python
# 10因子交叉截面动量
factor_returns = compute_10_factors(universe)  # 价值/动量/质量/低波/规模等
momentum_scores = factor_returns.rolling(21).mean()  # 1个月回看
selected_factor = momentum_scores.idxmax(axis=1)
portfolio = rebalance_to_factor(selected_factor)
```

#### S2: ETF 动量轮动 (Multi-Factor Cross-Sectional)

| 属性 | 内容 |
|------|------|
| **来源** | zhangsensen/etf-rotation-strategy (生产级实盘，胜率83.3%) |
| **逻辑** | 49只ETF(41 A股+8 QDII)，23因子排名，选Top2，Exp4滞后(持仓≥9天) |
| **回看期** | 60/120/252日加权 |
| **夏普比率** | **1.38** (VEC验证/2.33 WFO) |
| **最大回撤** | 10.8% (BT回测) |
| **胜率** | **83.3%** |
| **实盘状态** | v8.0已封版，生产环境中运行 |
| **风险门控** | 波动率分位 → 仓位缩放 100%→70%→40%→10% |

**23因子体系**:
```
OHLCV因子(17个): 动量(1/3/6/12月)、波动率(20/60日)、夏普(60/120/252日)、
                换手率、最大回撤、RSI、MACD、ATR比率、量比

非OHLCV因子(6个): 规模因子、费率因子、跟踪误差、溢价率、折价率、流动性
```

#### S3: 双动量 (Dual Momentum — GEM)

| 属性 | 内容 |
|------|------|
| **来源** | Antonacci (2017) "Risk Premia Harvesting Through Dual Momentum" |
| **逻辑** | 绝对动量: 过去12月是否跑赢国债？否→全转债券。相对动量: 选表现最好的指数 |
| **年化额外收益** | +440bp vs S&P 500 (1950-2018) |
| **熊市平均** | GEM +3.6% vs S&P 500 -37% |
| **牛市平均** | GEM +289.9% vs S&P 500 +214.9% |

#### S4: 行业动量轮动 (Correlation-Consolidated)

| 属性 | 内容 |
|------|------|
| **来源** | Boubaker, Du & Liu (2022) Journal of Asset Management |
| **逻辑** | 申万行业分类，6月形成/6月持有，相关系数0.75合并行业 |
| **形成/持有** | 6个月/6个月 |
| **月收益** | 4.8% (t统计量显著于1%) |
| **夏普比率** | **1.16**（相关系数整合后，原始0.71） |
| **关键细节** | 跳过二月（春节反转效应污染信号） |

#### S5: 52周新高动量

| 属性 | 内容 |
|------|------|
| **来源** | Lan & Truong (2024) |
| **逻辑** | 买入过去52周创过新高的指数ETF |
| **月收益** | 0.28% (中国A股) |
| **特点** | 与中国特有散户行为相关，Fama-French五因子后仍显著 |

---

### 3.1.2 均值回归与价值类（5个策略）

#### S6: PE/PB 估值分位带 ⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| **逻辑** | PE < 25%分位 → 分批建仓; PE > 75%分位 → 分批减仓 |
| **适用指数** | CSI 300, CSI 500, 创业板指 |
| **当前CSI 300** | PE 14.4 (62.2%分位) → 偏贵，减持或持有 |
| **买入阈值** | CSI 300 PE < 10.5, PB < 1.2 |
| **卖出阈值** | CSI 300 PE > 14.5, PB > 1.7 |
| **胜率** | ~60-70% 长期，价值因子周期性(3-4年) |

#### S7: RSI均值回归 + 固收 ⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| **来源** | 中信建投2026年6月低波固收+报告 |
| **逻辑** | RSI(14) < 35 买入指数ETF + RSI > 70 卖出，剩余时间持有债券 |
| **年化收益** | **6.00%** (10%沪深300 + 10%创业板动量 + 80%中债) |
| **波动率** | 2.88% |
| **最大回撤** | **-2.64%** |
| **胜率** | 70%+ (震荡市实测) |

**增强版 — 复合打分**:
```
总分 = RSI(20)得分 × 30% + 价格偏离MA得分 × 35% + 布林带偏离得分 × 35%
低分(≤25)买入 + 高分(≥75)卖出
年化: 9.32% | 最大回撤: -11.7% | 夏普: 0.660
```

#### S8: 红利低波 + 股息率择时 ⭐⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| **逻辑** | 中证红利指数(PE 7.16, 27%分位 → 低估)，股息率5.12% vs 国债利差2.83%(88.5%分位) |
| **累计超额** | 2021-2025 红利指数 +31.4% vs CSI 300 |
| **2021**: +35.04% | **2022**: +16.30% | **2023**: +11.94% | **2024**: -7.37% |
| **胜率** | 80%年度正超额 |

#### S9: 布林带 + RSI 复合震荡策略

| 属性 | 内容 |
|------|------|
| **参数** | 布林带周期20日，标准差2.0；RSI(14) < 30买 > 70卖 |
| **止损** | 跌破下轨后继续跌5% → 强制止损 |
| **胜率** | **68-70%** （震荡市实测，叩富网实盘平台） |

#### S10: 网格交易（震荡标的）

| 属性 | 内容 |
|------|------|
| **标的** | 沪深300ETF (510300) |
| **参数** | 间距5%、5层网格、等额买入 |
| **年化收益** | **6.17%** |
| **最大回撤** | **-12.3%** (vs 买入持有 -28.5%) |
| **胜率** | **68%** |
| **回测期间** | 2023-2026，44次交易 |

**网格适宜度筛选公式（审计增强版 — Hurst 一票否决）**:
```
适宜度 = ATR振幅(30%) + 波动率(25%) + ADX(20%) + 流动性(15%) + Hurst适应性(10%)

其中 Hurst适应性:
  Hurst < 0.4  → 10分  (强均值回归 → 网格天堂)
  0.4 ≤ H < 0.55 → 5分  (随机游走 → 网格勉强可行)
  0.55 ≤ H < 0.6 → 2分  (弱趋势 → 网格风险上升)
  Hurst ≥ 0.6  → 一票否决 (单边趋势确立 → 🛑 禁止开启网格)

≥70分 + Hurst < 0.6 = 非常适合
60-70分 + Hurst < 0.6 = 基本适合
<60分 或 Hurst ≥ 0.6 = 🛑 不适合（破网风险极高）

网格间距 = 年化波动率 × 1.5 / √252
```

**为什么 Hurst 一票否决至关重要**:
网格交易唯一致命的敌人是"单边不回头的破网行情"。Hurst 指数 > 0.6 意味着市场已进入趋势状态（如 2024 年 9 月暴涨），此时开网格 = 单边被突破后浮亏无限累积。Hurst > 0.6 时，必须强制切换到趋势跟踪策略。

---

### 3.1.3 因子轮动与宏观驱动类（4个策略）

#### S11: 宏观四状态轮动 (美林时钟中国版)

| 属性 | 内容 |
|------|------|
| **驱动指标** | PMI(阈值50) + CPI(阈值3%) |
| **四状态** | 衰退→债券; 复苏→股票; 过热→商品; 滞涨→现金 |
| **来源** | 海通证券/中银证券 验证 |
| **年化超额** | 4选2策略: **+16.1%** | 信息比: 1.78 | 月度胜率: **66%** |
| **4选1策略** | 年化超额: **+21%** | 信息比: 1.25 | 月度胜率: **60%** |

#### S12: 风格轮动 (大盘/小盘 × 价值/成长)

| 属性 | 内容 |
|------|------|
| **逻辑** | SVR机器学习模型通过47个宏观指标预测风格因子多空方向 |
| **再平衡** | 季度，基于风格预测对备选指数打分 |
| **胜率** | 季度胜率 **63.6%** | 年化超额: **+6.52%** |

#### S13: ETF 低波轮动

| 属性 | 内容 |
|------|------|
| **逻辑** | 选择过去60日波动率最低的3只ETF等权配置 |
| **年化收益** | **12.77%** |
| **最大回撤** | **-8.81%** |
| **夏普** | **1.06** |
| **来源** | ezquant GitHub 开源验证 |

#### S14: 春季效应策略 ("红包行情" / "春季躁动")

| 属性 | 内容 |
|------|------|
| **逻辑** | 春节前建仓小盘/创业板ETF → 节后20交易日持有 |
| **上升概率** | **80%** (2006-2025 20年数据) |
| **中位数收益** | **+9.45%** |
| **三大驱动因子** | ① 流动性释放（春节后资金回流） ② "两会"政策预期 ③ ★ **公募基金跨年排名战后"风格调仓"** |
| **策略增强** | AI 辩论时可作为加分定性因子 |

**★ 审计补充 — 公募基金调仓结构因子（定性洞察）**:
```
春季行情的一个深层机制:
12月底公募年度排名战结束 → 1月基金经理大规模调仓:
  - 获利了结去年重仓股（消费/蓝筹）
  - 重新布局新年度主线（科技/成长/小盘弹性标的）
  - 释放出的资金涌入小盘成长 → 推升创业板/中证1000

AI 辩论时将此因子作为定性权重:
  "12月排名战后基金经理调仓 → 小盘资金流入概率 +15%"
  结合历史日历效应 (80% 胜率) + 流动性释放 →
  CIO 的综合结论获得更高定性置信度
```

---

### 3.1.4 中国特色因子策略（3个策略）

#### S15: 国企改革红利 ("中特估")

| 属性 | 内容 |
|------|------|
| **逻辑** | 股息率 + 北向因子 + 主力资金 + 财务因子 复合选股；国企88组合 |
| **年化超额(全A)** | +3.65% | **年化超额(国企等权)**: +5.42% |
| **来源** | 华鑫证券量化国企88组合策略 |
| **当前状态** | 2024年国企市值管理改革政策催化 |

#### S16: 北向资金追随策略

| 属性 | 内容 |
|------|------|
| **逻辑** | 北向资金连续3日净流入 → 跟随买入；连续3日净流出 → 减仓 |
| **数据源** | 沪深港通每日披露 |

#### S17: 量化打板概率模型

| 属性 | 内容 |
|------|------|
| **逻辑** | 逐笔委托数据 → 推算封涨停概率 → 隔日溢价预期2-3% |
| **注意** | 2026年新监管规则，需标记为高风险策略 |
| **适用** | 仅限极短期操作，不推荐普通投资者 |

---

### 3.1.5 组合策略（3个策略）

#### S18: 三层防御型组合

| 资产 | 权重 | 策略 |
|------|------|------|
| 中证红利ETF | 40% | 股息率择时 |
| 中债7-10年 | 40% | 持有到期 |
| 黄金ETF | 20% | 趋势跟踪(MA200) |
| **年化收益**: 8-10% | **最大回撤**: <8% | **夏普**: 0.8-1.0 |

#### S19: 进取型ETF组合

| 资产 | 权重 | 策略 |
|------|------|------|
| CSI 300 ETF | 30% | 因子动量 |
| 创业板ETF | 20% | 趋势跟踪 |
| 行业ETF轮动 | 30% | 行业动量(6,6) |
| 黄金ETF | 10% | 风险平价 |
| 现金 | 10% | 机会储备 |
| **年化收益**: 12-15% | **最大回撤**: <20% | **夏普**: 0.9-1.2 |

#### S20: AI增强组合

| 方式 | 逻辑 |
|------|------|
| 基础配置 | S19 进取型组合 |
| AI叠加 | 多智能体每周末辩论 → 调整仓位 ±20% |
| 风格偏向 | CIO根据宏观信号：risk_on → 加仓成长/动量 | risk_off → 加仓红利/债券 |
| **年化收益(模拟)**: 15-18% | **最大回撤**: <15% | **AI提升**: +3-5%年化 |

---

### 3.2 策略性能排名

| 排名 | 策略 | 年化收益 | 夏普 | 最大回撤 | 胜率 | 可行性 |
|------|------|---------|------|---------|------|--------|
| 🥇 | **因子动量轮动** | 9.91% | 1.15 | -12.4% | 65% | ⭐⭐⭐⭐ |
| 🥈 | **ETF动量轮动** | 53.9%(OOS) | 1.38 | 10.8% | **83.3%** | ⭐⭐⭐⭐⭐ |
| 🥉 | **行业动量(相关整合)** | — | **1.16** | — | — | ⭐⭐⭐ |
| 4 | **ETF低波轮动** | 12.77% | 1.06 | -8.81% | — | ⭐⭐⭐⭐⭐ |
| 5 | **宏观4选2轮动** | +16.1%超额 | 1.78 IR | — | **66%** | ⭐⭐⭐ |
| 6 | **RSI均值回归+固收** | 6.00% | — | **-2.64%** | 70%+ | ⭐⭐⭐⭐⭐ |
| 7 | **红利低波择时** | 9.32% | 0.660 | -11.7% | — | ⭐⭐⭐⭐⭐ |
| 8 | **网格交易(震荡市)** | 6.17% | — | -12.3% | **68%** | ⭐⭐⭐⭐ |
| 9 | **春季效应** | +9.45%(中位数) | — | — | **80%** | ⭐⭐⭐⭐ |
| 10 | **三层防御组合** | 8-10% | 0.8-1.0 | <8% | — | ⭐⭐⭐⭐⭐ |

---

## <a name="4"></a>4. 回测引擎架构

### 4.1 三层验证体系

参考 zhangsensen/etf-rotation-strategy 生产级验证流水线 + Lopez de Prado过拟合检测：

```
Layer 1: WFO (Walk-Forward Optimization)  ~2分钟
  ├── 12,597 因子组合快速筛选
  ├── 复合评分: 收益40% + 夏普30% + 回撤30%
  ├── IC衰减门控 & 因子方向稳定性检测
  └── 输出: Top-5 参数组合 → 进入Layer 2

Layer 2: VEC (Vectorized Backtest)  ~5分钟
  ├── Numba @njit 编译的精确浮点份额模拟
  ├── 完整滞后状态机 (Exp4 hysteresis)
  ├── 成本模型 (A股2bp, QDII 5bp)
  └── 输出: 最优参数组合 → 进入Layer 3

Layer 3: BT (Event-Driven Backtest)  ~30-60分钟
  ├── Backtrader 事件驱动引擎
  ├── 整数手约束 + 真实基金限制
  ├── T+1开盘价执行
  └── 输出: 生产级绩效（与实盘差异 <2%）
```

### 4.2 过拟合检测

```
Layer 4: CPCV (Combinatorial Purged Cross-Validation)
  ├── C(N,K) 训练/测试路径
  ├── Purge + Embargo 防信息泄漏
  ├── 输出: OOS分布（非单条曲线）
  └── PBO (Probability of Backtest Overfitting):
      策略在IS最优但在OOS低于中位数的概率
      PBO < 10% → OK, PBO > 30% → 拒绝

Layer 5: DSR (Deflated Sharpe Ratio)
  ├── 考虑 M=12,597 次试验的多重测试修正
  ├── DSR > 1.0 (95%置信) → 统计显著
  └── 最小回测长度: T_min > (SR²/4)(1+SR²/4) × 252天

Layer 6: Monte Carlo 鲁棒性检验
  ├── Block Bootstrap (保持自相关)
  ├── 1000次重采样 → 5%/95%置信带
  ├── 验证: Sharpe稳定性 / MaxDD分布 / 收益曲线一致性
  └── ⚠️ 关键发现: Bootstrap的MaxDD置信区间过于乐观 7-23%
      实际回撤 = 名义回撤 × 1.5-5x
```

### 4.3 中国公募基金特有交易规则（审计补充 — 实盘生死线）

以下规则是外部审计指出的原方案最大盲区。大部分开源回测引擎基于美股"理想化连续交易"假设设计，直接套用在**场外公募基金**上将导致回测收益虚高 30-50%。

#### 4.3.1 场内 ETF vs 场外公募 — 双通道交易规则

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from datetime import time


class FundChannel(str, Enum):
    """基金交易通道 — 两条完全不同的流水线"""
    ETF_ON_EXCHANGE = "ETF_ON_EXCHANGE"       # 场内 ETF (二级市场买卖)
    OTC_OPEN_END = "OTC_OPEN_END"             # 场外开放式基金 (申购/赎回)
    OTC_ETF_FEEDER = "OTC_ETF_FEEDER"         # ETF 联接基金 (场外)
    QDII = "QDII"                              # 跨境 QDII 基金


@dataclass(frozen=True)
class FundTradingProfile:
    """每只基金的交易特征 — 从 AKShare 基金合同中提取"""
    fund_code: str
    channel: FundChannel
    # 场外规则
    cutoff_time: time = time(15, 0)           # 15:00 生死截点
    nav_publish_delay_hours: int = 6           # 净值通常在 20:00-22:00 披露
    # 赎回费阶梯 (2026 新规)
    redemption_fee_schedule: dict = None       # { "<7d": 0.015, "7-30d": 0.01, ... }
    # 资金交收延迟
    settlement_delay_days: int = 0             # ETF=0, OTC=4, QDII=7-10
    # 容量限制
    max_subscription_per_day_yuan: Optional[float] = None  # 大额申购限额
    is_suspended: bool = False                 # 暂停申购/赎回
```

#### 4.3.2 15:00 申赎截点验证器

```python
class OrderCutoffValidator:
    """
    场外公募基金申赎截点 — 回测中最容易出错的规则

    规则:
    - T 日 15:00 前申请 → 按 T 日收盘净值成交
    - T 日 15:00 后申请 → 顺延至 T+1 交易日，按 T+1 净值成交

    AI 开发陷阱:
    如果多智能体辩论系统在盘后 (20:00-22:00) 生成推荐，
    回测引擎绝不能假设能拿到当日成交净值，必须强制以 T+1 日净值撮合。
    """

    CUTOFF = time(15, 0)

    def resolve_execution_date(self, signal_time, is_trading_day_fn):
        """
        输入: 信号生成时间 (datetime)
        输出: 实际以哪天净值成交 (date)

        例如:
        - 6月26日 14:30 的信号 → 以 6月26日净值成交 ✅
        - 6月26日 15:01 的信号 → 顺延至 6月27日成交 ⚠️
        - 6月26日 22:00 (净值披露后/AI 辩论完成) → 必须顺延至 6月27日 🚨
        """
        if signal_time.time() <= self.CUTOFF and is_trading_day_fn(signal_time.date()):
            return signal_time.date()  # T 日
        else:
            return self._next_trading_day(signal_time.date(), is_trading_day_fn)  # T+1

    def validate_backtest_order(
        self, signal_time, nav_publish_time, execution_date
    ) -> bool:
        """
        检查回测中是否存在 "前瞻偏差" (look-ahead bias):
        信号时间 > 15:00 且使用了当日净值 → 拒绝
        """
        if signal_time.time() > self.CUTOFF:
            if execution_date == signal_time.date():
                return False  # 🚨 错误: 15:00 后的信号用了当日净值
        if signal_time < nav_publish_time and execution_date == signal_time.date():
            return False  # 🚨 错误: 净值披露前不可能知道当日净值
        return True
```

#### 4.3.3 2026 公募销售新规阶梯赎回费计算器

```python
class RedemptionFeeCalculator:
    """
    2026 年修订的《公开募集证券投资基金销售费用管理规定》

    赎回费全额计入基金财产，是高频轮动策略的 Alpha 杀手。

    阶梯费率表 (场外普通公募):
    ┌──────────────────────┬──────────────┬─────────────────────┐
    │ 持有天数 D           │ 法定最低费率 │ 说明                │
    ├──────────────────────┼──────────────┼─────────────────────┤
    │ D < 7 天             │ ≥ 1.5%       │ 惩罚性费率          │
    │ 7 ≤ D < 30 天        │ ≥ 1.0%       │ 2026 新规新增       │
    │ 30 ≤ D < 180 天      │ ≥ 0.5%       │ 持有不满半年        │
    │ D ≥ 180 天           │ 0% (多数)    │ 鼓励长期持有        │
    └──────────────────────┴──────────────┴─────────────────────┘

    场内 ETF 在二级市场买卖: 0 赎回费，仅收券商佣金
    """

    # 法定最低赎回费率 (场外普通公募)
    STATUTORY_MIN_FEE = {
        (0, 7): 0.015,       # < 7 天
        (7, 30): 0.010,      # 7-30 天 (2026 新规)
        (30, 180): 0.005,    # 30-180 天
        (180, float("inf")): 0.0,  # ≥ 180 天
    }

    # 豁免类别 (允许合同另行约定，通常费率更低)
    EXEMPT_CATEGORIES = {"INDEX_FUND", "SPECIAL_BOND", "MONEY_MARKET"}

    def calculate(self, fund_profile, holding_days, redemption_amount):
        """
        计算赎回费

        Args:
            fund_profile: 基金交易特征
            holding_days: 持仓天数
            redemption_amount: 赎回金额

        Returns:
            赎回费 (元), 赎回费率 (%)
        """
        # 场内 ETF: 不收取赎回费
        if fund_profile.channel == FundChannel.ETF_ON_EXCHANGE:
            return 0.0, 0.0

        # 豁免类别: 使用合同约定费率 (从 AKShare 提取)
        if fund_profile.fund_category in self.EXEMPT_CATEGORIES:
            contract_fee = fund_profile.redemption_fee_schedule or {}
            rate = self._lookup_fee(contract_fee, holding_days)
        else:
            # 普通场外公募: 法定最低费率
            rate = self._lookup_fee(self.STATUTORY_MIN_FEE, holding_days)

        return redemption_amount * rate, rate

    def _lookup_fee(self, schedule, holding_days):
        """查找持有天数对应的费率"""
        for (lo, hi), rate in sorted(schedule.items()):
            if lo <= holding_days < hi:
                return rate
        return 0.0

    def enforce_min_hold(self, holding_days, fund_profile) -> str:
        """
        风控: 短持惩罚熔断

        如果策略建议持有 <7 天且为场外公募:
        → 警告: "此交易将被收取 1.5% 惩罚性赎回费"
        → 强制标记为 HIGH_COST 信号
        """
        if fund_profile.channel != FundChannel.ETF_ON_EXCHANGE:
            if holding_days < 7:
                return "🛑 REJECT: 持有不足7天, 赎回费 1.5% 将吞噬利润"
            elif holding_days < 30:
                return "⚠️ WARN: 持有不满30天, 赎回费 1.0%"
        return "PASS"
```

#### 4.3.4 资金交收延迟状态机 (CashLockManager)

```python
from collections import deque
from dataclasses import dataclass, field


@dataclass
class LockedCash:
    """一笔被冻结的资金"""
    amount: float
    unlock_date: int          # 解冻的交易日序号
    fund_type: str
    redeem_date: int          # 赎回发生的交易日序号


class CashLockManager:
    """
    资金交收延迟 — 消除回测中的"无限免费过桥资金"

    真实世界微观结构 (2026):
    - 场内 ETF 卖出: 资金当日可用于买入其他 ETF (T+0 可用)，T+1 可取
    - 场外普通公募赎回: T+3 到 T+5 交易日到账
    - ETF 联接基金赎回: T+2 到 T+3 交易日到账
    - QDII 基金赎回: T+6 到 T+10 交易日到账 (跨时区+外汇结汇)

    如果不模拟此延迟 → 回测中相当于开了"无限零息过桥资金"的外挂
    """

    def __init__(self, initial_cash: float):
        self.available_cash = initial_cash
        self.locked_queue: deque[LockedCash] = deque()

    def lock(self, amount, fund_type, current_day):
        """
        赎回发生时立即锁定资金

        Args:
            amount: 赎回金额
            fund_type: 基金类别
            current_day: 当前交易日序号
        """
        delay = self._settlement_delay(fund_type)
        unlock_day = current_day + delay

        self.locked_queue.append(LockedCash(
            amount=amount,
            unlock_date=unlock_day,
            fund_type=fund_type,
            redeem_date=current_day,
        ))
        self.available_cash -= amount

    def unlock_daily(self, current_day):
        """
        每日结算: 释放到期资金

        应在每个交易日开始时调用
        """
        unlocked = 0.0
        while self.locked_queue and self.locked_queue[0].unlock_date <= current_day:
            cash = self.locked_queue.popleft()
            unlocked += cash.amount

        self.available_cash += unlocked
        return unlocked

    def get_buying_power(self, current_day):
        """获取当日实际可用购买力"""
        return self.available_cash

    def _settlement_delay(self, fund_type_str):
        """资金交收到账天数（交易日）"""
        DELAYS = {
            "ETF_STOCK": 0,         # 场内 ETF，当日可复用
            "EQUITY_OTC": 4,        # 场外普通公募，T+3 到 T+5
            "ETF_FEEDER": 3,        # ETF 联接基金，T+2 到 T+3
            "QDII": 8,              # QDII 基金，T+6 到 T+10 (取中值)
            "MONEY_MARKET": 1,      # 货币基金，T+1 到 T+2
        }
        return DELAYS.get(fund_type_str, 4)  # 默认保守估计

    def get_locked_summary(self) -> dict:
        """获取资金冻结概览"""
        if not self.locked_queue:
            return {"locked_total": 0, "pending_items": 0}

        return {
            "locked_total": sum(c.amount for c in self.locked_queue),
            "pending_items": len(self.locked_queue),
            "next_unlock_day": self.locked_queue[0].unlock_date,
            "max_lock_days": max(c.unlock_date - c.redeem_date for c in self.locked_queue),
        }
```

#### 4.3.5 大额申购限额与巨额赎回检测

```python
class CapacityGate:
    """
    容量限制

    1. 大额申购限额: QDII 外汇额度、明星基金经理控规模
       → 单日限购 100-1000 元常见
    2. 巨额赎回: 单日净赎回 > 基金总份额 10%
       → 管理人有权限拒绝或顺延
    """

    def check_subscription(self, fund_profile, desired_amount):
        """检查申购是否可行"""
        if fund_profile.is_suspended:
            return False, "暂停申购"

        max_amount = fund_profile.max_subscription_per_day_yuan
        if max_amount and desired_amount > max_amount:
            return False, f"超过单日申购限额 {max_amount:.0f} 元"
            # 策略建议: 截断至限额而非完全拒绝
            # allowed = max_amount

        return True, "OK"

    def check_mass_redemption(self, redemption_amount, fund_total_size):
        """巨额赎回风险检测"""
        ratio = redemption_amount / fund_total_size if fund_total_size > 0 else 0
        if ratio > 0.10:
            return False, f"触发巨额赎回线 ({ratio:.1%} > 10%)"
        return True, "OK"
```

#### 4.3.6 场内 ETF 伪代码（对比参照）

```python
# 场内 ETF 交易规则 — 保持原有设计
ETF_RULES = {
    "t_plus_1": True,          # T日信号 → T+1日开盘价执行
    "price_limit_main": 0.10,  # 主板 ±10%
    "price_limit_gem": 0.20,   # 创业板/科创板 ±20%
    "price_limit_st": 0.05,    # ST ±5%
    "min_lot": 100,            # 最小交易单位100股
    "stamp_duty_sell": 0.0005, # 卖出印花税0.05% (2023年降)
    "commission": 0.00025,     # 佣金万2.5
    "transfer_fee": 0.00001,   # 过户费0.001% 双向
    "no_short": True,          # 零售不可卖空
    "slippage_model": "bid_ask_spread + sqrt_impact",
    "funds_available_same_day": True,   # 卖出后当日可复用买其他ETF
}
```

### 4.4 绩效指标全套

| 类别 | 指标 | 阈值 |
|------|------|------|
| **收益** | 年化收益、累计收益、超额收益 | — |
| **风险调整** | Sharpe (>1.0好) / Sortino / Calmar (>1.0可接受) | — |
| **回撤** | 最大回撤、回撤持续时间、恢复时间、水下曲线 | MDD < 20% |
| **胜率** | 胜率、盈亏比、Profit Factor (>1.5) | 胜率 > 55% |
| **尾部风险** | VaR(95%), CVaR(95%), Pain Index, Ulcer Index | CVaR < 15% |
| **一致性** | SQN (>1.0可接受, >2.0优秀), Omega比率 | SQN > 1.5 |
| **基准相对** | 跟踪误差、信息比率(>0.5好)、主动份额、捕获比率 | IR > 1.0 |

---

## <a name="5"></a>5. 策略切换机制

### 5.1 市场状态识别系统

参考 alpha-forge 的 MarketRegimeDetector 架构 + HMM + GARCH 学术成果：

```
输入特征 (每日计算):
├── 收益率 (1d/5d/20d)
├── 波动率 (20d rolling, GARCH预测)
├── RSI(14), ADX(14), Choppiness Index
├── Hurst Exponent (趋势/均值回归判断)
├── 相关系数矩阵 (申万行业间)
├── 宏观: PMI, CPI, 利率, 北向资金流
└── 流动性: 成交额/5日均量比, 融资余额变化

状态分类 (3-5状态HMM + 规则补充):
├── REGIME_TRENDING_UP:    ADX > 25 + 价格 > MA60 + Hurst > 0.6
├── REGIME_TRENDING_DOWN:  ADX > 25 + 价格 < MA60 + Hurst > 0.6
├── REGIME_SIDEWAYS:       ADX < 20 + Choppiness > 61.8
├── REGIME_HIGH_VOL:       GARCH预测波动 > 历史80%分位
└── REGIME_CRISIS:         波动率突破3σ + 行业相关性 > 0.8

状态更新频率: 每周重检(避免过度切换)
```

### 5.2 状态-策略映射表

| 市场状态 | 推荐策略 | 仓位 | 止损 |
|----------|---------|------|------|
| **TRENDING_UP** | 因子动量 + ETF动量轮动 + 行业动量 | 100% | 10% trailing |
| **TRENDING_DOWN** | 红利低波 + RSI均值回归(反向) + 国债 | 30% | 5% stop |
| **SIDEWAYS** | 网格交易 + 布林带RSI + 低波轮动 | 70% | 8% stop |
| **HIGH_VOL** | 波动率目标(降仓) + 红利防御 | 40% | 5% stop |
| **CRISIS** | 国债100% 或 黄金+国债50/50 | 0-20% | 3% hard stop |

### 5.3 动态仓位分配

```python
def allocate_capital(market_regime, strategies, capital):
    """
    基于市场状态的策略仓位分配
    """
    eligible = [s for s in strategies if market_regime in s.regimes]
    scores = [s.rolling_sharpe * s.regime_alignment(market_regime) for s in eligible]

    # HMM转移概率 → 软分配（防过度切换）
    weights = softmax(scores) * regime_confidence(market_regime)

    # 换手率控制: 单次调仓 ≤ 30%
    weights = turnover_limit(weights, previous_weights, max_turnover=0.30)

    # 单策略上限: 50%
    weights = np.clip(weights, 0, 0.50)

    return normalize(weights) * capital * position_scale(market_regime)
```

### 5.4 换手率控制（防过度切换）

基于 LoT (Low-turnover Control, Lewin 2022) 和 5%漂移带最优阈值:

```python
class TurnoverController:
    """
    LoT: 动态阈值 = u分位数, u = 1 - 状态转移概率
    转移概率高 → 阈值低 → 更积极调仓
    转移概率低 → 阈值高 → 减少不必要调仓
    """
    def should_rebalance(self, target_weights, current_weights, transition_prob):
        turnover = sum(abs(target_weights - current_weights)) / 2
        u = 1 - transition_prob
        threshold = np.percentile([turnover], u * 100)
        return turnover > threshold and turnover > 0.05  # 最少5%差异才触发
```

---

## <a name="6"></a>6. 投前验证框架

### 6.1 五层验证门控

策略必须全部通过以下五层验证才能进入实盘：

```
Layer 1: 回测统计显著性
  ├── DSR > 1.0 (95%置信水平)
  ├── PBO < 20%
  ├── Min Track Record > 3年 (252天×3)
  └── Out-of-Sample Sharpe >= 0.7 × In-Sample Sharpe (无严重过拟合)

Layer 2: 参数稳定性
  ├── 参数扰动 ±20% → 绩效波动 < 30%
  ├── 至少2个替代参数窗口产生一致方向
  └── 参数数量 < √(训练观测数)

Layer 3: 多制度压力测试
  ├── 历史场景回放: 2015崩盘(-43%), 2018贸易战(-25%), 2020新冠(-15%)
  ├── 假设冲击: 波动率×2, 相关性→1, 利率+2%
  ├── 每个场景: Sharpe > 0 且 MaxDD < 历史×1.2
  └── 所有场景通过率 ≥ 80%

Layer 4: 成本敏感性
  ├── 基准成本 (A股2bp单边) → 净Sharpe
  ├── 成本×2 → 净Sharpe 下降 < 30%
  ├── 成本×5 → 净Sharpe 仍 > 0
  └── 结论: 策略必须在极端成本下仍不亏损

Layer 5: 容量上限
  ├── 策略容量 = min(日成交额×0.1%, AUM×1%)
  ├── 容量在个人投资规模下(100万)无限制 → PASS
  └── 超出容量 → 标记不可扩展
```

### 6.2 模拟交易（纸上验证）

在通过五层验证后，策略进行 **30个交易日** 的模拟交易：

```
纸上交易规则:
├── 使用真实T日收盘信号 + T+1日开盘价执行
├── 包含真实费用(佣金/印花税/过户费)
├── 模拟滑点: bid-ask spread + sqrt(Q/V)冲击
├── 整数手约束 + 涨跌停板无法成交
├── 每日产生成交报告 + PnL快照
└── 30日后绩效与回测偏差 < 5% → 进入实盘
```

### 6.3 实盘监控

策略上线后的持续监控：

```
每日检查:
├── Rolling 20日Sharpe vs 回测Sharpe: 偏差 < 50%
├── 单日回撤 < -3% → 黄色预警
├── 滚动5日回撤 < -8% → 红色预警 → 人工复核

每周检查:
├── Rolling IC衰减检测 (Page-Hinkley test)
├── 参数漂移检测 (贝叶斯在线断点检测)
└── 制度对齐: 当前市场状态 vs 策略设计状态

月度检查:
├── 完整绩效归因 (Brinson 配置效应 + 选股效应)
├── 因子暴露检查 (CH-3模型Alpha仍为正?)
└── 容量上限重新评估
```

---

## <a name="7"></a>7. 风险管理体系

### 7.1 四层风控架构

```
Layer 1: 仓位层 — 单标的层面
├── 固定比例: 单标的最多20%仓位
├── 波动率目标: annualized_vol ≤ 15% → position = capital × 0.15 / realized_vol
├── Kelly分数: position = fractional_kelly(win_rate, win_loss_ratio) × capital
└── ATR止损: stop_price = entry_price - 2 × ATR(14)

Layer 2: 策略层 — 单策略层面
├── 最大回撤止损: 滚动MaxDD > 15% → 策略暂停
├── 连续亏损: 连续5次止损 → 策略暂停
├── VaR限制: 日VaR(95%) ≤ 组合的2%
└── 换手率上限: 月换手 ≤ 100%

Layer 3: 组合层 — 多策略层面
├── 风险预算: 每策略风险贡献 = (w_i × σ_i) / σ_portfolio = 目标比例
├── HRP聚类: 策略相关性聚类 → 组内等风险贡献
├── 最大组合回撤: rolling MaxDD > 20% → 总体减仓50%
└── 相关性熔断: 跨策略相关性 > 0.8 → 降杠杆

Layer 4: 极端风险层
├── 尾部风险对冲: 组合CVaR(99%) > 历史2σ → 自动买Put/减仓
├── 流动性枯竭: 日成交额 < 20日均量×0.3 → 禁止交易
├── 黑天鹅熔断: CSI 300单日跌>7% → 全部平仓
└── 政策风险: 行业性政策突变 → 24h内出清受影响标的
```

### 7.2 仓位计算公式汇总

```python
# 方法1: 固定分数
position = capital * risk_per_trade / abs(entry - stop)

# 方法2: 波动率目标
position = capital * target_annual_vol / (realized_vol * sqrt(252))

# 方法3: 分数Kelly
f_star = win_rate - (1 - win_rate) / win_loss_ratio
fractional_f = f_star * 0.25  # 25% Kelly（生产默认值）
position = capital * fractional_f

# 方法4: ATR动态调整
risk_per_share = 2 * ATR(14)
position_shares = capital * risk_per_trade / risk_per_share
```

### 7.3 风险度量全套

| 指标 | 公式/说明 | 预警阈值 | 熔断阈值 |
|------|----------|---------|---------|
| **VaR(95%)** | 历史模拟法5%分位 | >2%组合 | >5%组合 |
| **CVaR(95%)** | VaR尾部均值 | >3%组合 | >7%组合 |
| **最大回撤** | 滚动峰值到谷底 | >10% | >20% |
| **回撤持续时间** | 峰值到恢复的天数 | >60天 | >120天 |
| **杠杆率** | 总敞口/净资产 | >1.0 | >1.5 |
| **集中度HHI** | Σ w_i² | >0.15 | >0.25 |
| **相关性** | 平均成对相关 | >0.6 | >0.8 |

---

## <a name="8"></a>8. 实施路线图

### Phase 1: 基础设施 — Headless Core 骨架 (Week 1-2)

```
T1.1  项目结构初始化 (uv/pyproject.toml, src/core/ 分层)
T1.2  统一服务层 API 入口 (src/core/api.py)
      → 所有对外接口仅返回 Dict/DataFrame，零 UI 依赖
T1.3  SQLite + DuckDB 数据库 Schema 设计 & CRUD 层
T1.4  AKShare 数据采集模块 (基金净值/ETF/指数)
T1.5  东方财富直接 API 备用采集
T1.6  数据缓存层 (本地 Parquet + 增量更新策略)
T1.7  基金合同解析模块
      → 自动提取: 赎回费率表、申购限额、交易通道(场内/场外/ETF联接/QDII)
T1.8  配置文件系统 (YAML, Pydantic 验证)
T1.9  CLI 入口点 (fund-research tui / web / update / analyze)
```

### Phase 2: 分析引擎 (Week 3-4)

```
T2.1  MyTT 技术指标模块 (KDJ/MACD/BOLL/RSI/MA)
T2.2  fincore 风险指标模块 (Sharpe/Sortino/Calmar/VaR/CVaR)
T2.3  jh-factors CH-3 因子模型集成
T2.4  绩效归因模块 (Brinson + 因子归因)
T2.5  8因子基金评分系统
T2.6  bardsai 金融情绪分析 + SnowNLP 备用
T2.7  新闻采集 & 情感标注流水线
```

### Phase 3: 回测引擎 — 中国规则核心 (Week 5-6)

```
T3.1  VectorBT 集成 (快速参数扫描)
T3.2  Backtrader 引擎 (事件驱动 + A股规则)
T3.3  ★ OrderCutoffValidator (15:00 申赎截点)
T3.4  ★ RedemptionFeeCalculator (2026 阶梯赎回费)
T3.5  ★ CashLockManager (资金交收延迟状态机)
T3.6  ★ CapacityGate (大额申购/巨额赎回限制)
T3.7  场内 ETF 交易规则 (t+1/涨跌停/佣金/印花税/滑点)
T3.8  AShareSlippageModel (Level-2 滑点建模)
T3.9  WFO 前向优化流水线
T3.10 CPCV 过拟合检测
T3.11 DSR 缩水夏普比率实现
T3.12 Monte Carlo 鲁棒性检验
T3.13 三层验证自动化 (WFO→VEC→BT)
```

### Phase 4: 策略库 (Week 7-8)

```
T4.1  因子动量轮动 (S1)
T4.2  ETF 多因子动量 (S2)
T4.3  双动量 GEM (S3)
T4.4  行业动量轮动 (S4)
T4.5  PE/PB 估值分位带 (S6)
T4.6  RSI 均值回归+固收 (S7)
T4.7  红利低波择时 (S8)
T4.8  布林带 RSI 复合 (S9)
T4.9  网格交易 + ★ Hurst 一票否决门控 (S10)
T4.10 宏观四状态轮动 (S11)
T4.11 ETF 低波轮动 (S13)
T4.12 春季效应 + ★ 公募调仓因子 (S14)
T4.13 国企红利 (S15)
T4.14 三层防御组合 (S18)
T4.15 进取型 ETF 组合 (S19)
```

### Phase 5: 市场状态与切换 (Week 9)

```
T5.1  HMM 3 状态市场分类器
T5.2  GARCH 波动率预测模块
T5.3  ADX/Hurst/Choppiness 趋势强度指示器
      → ★ Hurst 指数同时供给 S10 网格策略一票否决
T5.4  状态-策略映射表
T5.5  动态仓位分配器
T5.6  换手率控制器 (LoT 算法)
T5.7  拥挤度监控 (CrowdingMonitor)
T5.8  每周状态重检定时器
```

### Phase 6: AI 决策引擎 (Week 10-11)

```
T6.1  多智能体框架 (LangGraph)
T6.2  Macro Strategist Agent + prompt
T6.3  Quant Analyst Agent + prompt
T6.4  Risk Officer Agent + prompt
T6.5  CIO Agent + 判决解析 + 6 项安全检查
T6.6  辩论协议 (Round 1 平行 → R2+ 交叉 → 收敛检测)
T6.7  信息隔离协议实现
T6.8  Isotonic Regression 置信度校准
T6.9  三阶段记忆系统 (Light/REM/Deep Sleep)
T6.10 8因子 + LLM 定性融合 → 最终推荐
T6.11 FactAnchoredAgent (强制数据引用)
T6.12 DualLLMValidator (双模型交叉验证)
T6.13 HallucinationAuditor (7/30天幻觉审计)
```

### Phase 7: TUI 终端界面 (Week 12)

```
T7.1  Textual App 框架搭建 (直接 import src.core.api)
T7.2  键盘命令面板 (Wind/Choice 风格键盘精灵)
T7.3  自选列表面板 (DataTable + 排序/筛选)
T7.4  K线图表面板 (plotext 集成)
T7.5  新闻面板 (标题+情感标签)
T7.6  推荐面板 (BUY/SELL 信号 + 因子得分 + LLM 调整 + 安全检查)
T7.7  多面板布局 (Grid 3×2)
T7.8  F5(行情)/F9(深度资料)快捷键体系
T7.9  ★ 双数字展示: 回测夏普 / 预期实盘夏普
T7.10 拥挤度指示器 (绿/黄/红)
T7.11 CJK 字体适配与对齐
T7.12 asyncio Worker 绑定 (防止 AI 辩论阻塞界面)
```

### Phase 8: Web 客户端 + 验证上线 (Week 13-14)

```
T8.1  FastAPI 服务搭建 (src/web/server.py)
      → 包装 src/core/api.py 为 REST JSON 接口
T8.2  WebSocket 实时推送 (AI 辩论进度 / 回测结果)
T8.3  Vue 3 / React 前端框架搭建
T8.4  ECharts K线图 + 绩效仪表盘
T8.5  五层验证门控自动化测试
T8.6  30 日纸上交易模拟
T8.7  实盘监控面板
T8.8  绩效归因报告自动生成
T8.9  因子衰减自动检测 (Page-Hinkley)
T8.10 ★ 幻觉审计月度排名面板
T8.11 衰减系数 β 自动更新机制
T8.12 文档 & 使用指南
```

---

## 附录A: 核心依赖清单

```txt
# 数据
akshare>=1.18.0
baostock>=0.8.8
efinance>=0.5.0

# 技术指标
pandas-ta>=0.4.0
# MyTT (单文件，直接复制)

# 风险分析
fincore>=0.7.0
jquantstats>=0.9.0
jh-factors>=0.1.9

# 回测
vectorbt>=0.27.0
backtrader>=1.9.0
pybroker>=1.4.0
skfolio>=0.4.0

# 情绪分析
snownlp>=0.12.0
jieba>=0.42.0
transformers>=4.40.0
torch>=2.0.0

# TUI
textual>=0.70.0
plotext>=5.2.0

# 工具
pandas>=2.2.0
numpy>=1.26.0
numba>=0.59.0
pydantic>=2.0.0
pyyaml>=6.0

# 数据库
# SQLite (内置)
duckdb>=1.0.0
pyarrow>=15.0.0

# LLM
openai>=1.50.0  # DeepSeek API兼容
# Ollama (本地部署，非Python包)
```

## 附录B: 参考开源项目

| 项目 | 最佳参考模块 |
|------|-------------|
| [zhangsensen/etf-rotation-strategy](https://github.com/zhangsensen/etf-rotation-strategy) | 三层验证+策略架构(⭐44) |
| [ailabx/alphalab](https://github.com/ailabx/alphalab) | 策略模板+Algo模块化组合(⭐945) |
| [alpha-forge](https://github.com/Liu-Ming-Yu/alpha-forge) | 市场状态检测+向量化回测(⭐167) |
| [BreadFree-Simu](https://github.com/FeiCoder/BreadFree-Simu) | 多智能体LangGraph(⭐149) |
| [mlfinpy](https://github.com/baobach/mlfinpy) | Lopez de Prado实现(⭐72) |
| [openInvest](https://github.com/longsizhuo/openInvest) | 4角色辩论委员会 |
| [FinAgent](https://github.com/helloJamest/FinAgent) | 6角色辩论+反射学习(⭐101) |
| [TradingAgents-astock](https://github.com/simonlin1212/TradingAgents-astock) | A股7分析师LangGraph(⭐431) |
| [Microsoft Qlib](https://github.com/microsoft/qlib) | AI量化+双集成模型(⭐40K+) |
| [vnpy/vnpy](https://github.com/vnpy/vnpy) | 事件驱动+CTA策略模板(⭐35K+) |
| [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) | 生产级基础设施(⭐10K+) |

---

## 附录C: 实施进度追踪 (v0.1.0 → v0.1.7)

**最后更新**: 2026-06-26 · **当前版本**: v0.1.8 · **测试**: 557/557 GREEN

### 总体进度: 52/89 (58%)

| Phase | 已实现 | 总数 | 进度 |
|-------|--------|------|------|
| 1: 基础设施 | 7 | 9 | 78% |
| 2: 分析引擎 | 4 | 7 | 57% |
| 3: 回测引擎 | 7 | 13 | 54% |
| 4: 策略库 | 14 | 15 | 93% |
| 5: 市场状态 | 5 | 8 | 63% |
| 6: AI 决策 | 5 | 13 | 38% |
| 7: TUI 终端 | 7 | 12 | 58% |
| 8: Web+验证 | 3 | 12 | 25% |

### 版本演进与外部审计修复记录

| 版本 | 修复项 | 测试数 |
|------|--------|--------|
| v0.1.0 | 初始发布: Headless Core + 中国规则引擎 + TUI/Web/CLI | 311 |
| v0.1.1 | 审计#1: 费率熔断/符号对齐/配置漂移/真空期/Kelly | 312 |
| v0.1.2 | 审计#2: 极端分数自残/假想现金/StrategyContext/工厂函数 | 313 |
| v0.1.3 | 审计#3: TUI CSS修复 + Web root定向 + sync def + Pydantic DTO | 313 |
| v0.1.4 | 审计#4: 骨架屏保护 + 真实数据绑定 + Web仪表盘 + 4新策略 + 风险指标/归因/CPCV+DSR | 434 |
| v0.1.5 | TUI Pilot自动化测试管线 + AI CI闭环脚本 | 435 |
| v0.1.6 | 冷启动自动注水 (on_ready hydration) + Pilot冷启动检测 | 435 |
| v0.1.7 | Web专业级分窗格大屏 (Vue3+Tailwind+ECharts) | 435 |
| v0.1.8 | 审计#5: 4计量硬伤(factor_model共线性/CPCV剪裁/DSR概率/soe_reform量纲) + CrowdingMonitor/LoT/CH-3 + 4新策略 | 557 |

### 已实现策略 (18/20)

| # | 策略 | 类别 | 文件 |
|---|------|------|------|
| S1 | 因子动量轮动 | 动量 | `momentum/factor_momentum.py` |
| S2 | ETF 多因子动量 | 动量 | `momentum/etf_momentum.py` |
| S3 | 双动量 GEM | 动量 | `momentum/dual_momentum.py` |
| S4 | 行业动量轮动 | 动量 | `momentum/sector_rotation.py` |
| S5 | 52周新高动量 | 动量 | `momentum/fiftytwo_week_high.py` |
| S6 | PE/PB 估值分位带 | 均值回归 | `mean_reversion/pe_pb_band.py` |
| S7 | RSI 均值回归+固收 | 均值回归 | `mean_reversion/rsi_fixed_income.py` |
| S8 | 红利低波择时 | 均值回归 | `mean_reversion/dividend_timing.py` |
| S9 | 布林带 RSI 复合 | 均值回归 | `mean_reversion/bollinger_rsi.py` |
| S10 | 网格交易+Hurst | 均值回归 | `mean_reversion/grid_hurst.py` |
| S11 | 宏观四状态轮动 | 因子轮动 | `factor_rotation/macro_rotation.py` |
| S13 | ETF 低波轮动 | 因子轮动 | `factor_rotation/low_vol_rotation.py` |
| S14 | 春季效应 | 因子轮动 | `factor_rotation/spring_festival.py` |
| S15 | 国企改革红利(中特估) | 中国特色 | `china_specific/soe_reform.py` |
| S18 | 三层防御组合 | 组合 | `portfolio/defensive.py` |
| S19 | 进取型ETF组合 | 组合 | `portfolio/aggressive.py` |

### 已知待推进项

- Phase 3: VectorBT/Backtrader 回测引擎集成
- Phase 5: HMM 分类器 / GARCH 模块
- Phase 6: LLM Agent 实现 (Macro/Quant/Risk/CIO LangGraph)
- Phase 8: Vue3 完整前端构建 / 30日纸上交易模拟
- Phase 2: fincore/jh-factors 外部库集成
- Phase 4: S12/S16/S17/S20 剩余策略

### TUI 稳定性保障体系

1. **骨架屏保护**: `on_mount` 建立列结构, `clear(columns=False)` 仅清行
2. **冷启动注水**: `on_ready` → `await action_refresh_portfolio()` 自动加载数据
3. **Pilot 管线**: 6项检查 (冷启动检测/时序真空点击/热键轰炸/输入攻击/数据咬合/布局)
4. **CI 闭环**: `run_ai_tui_ci.py --loop` 自动修复流水线

### 已知待推进项

- Phase 3: VectorBT/Backtrader 回测引擎集成
- Phase 5: HMM 分类器 / GARCH 模块
- Phase 6: LLM Agent 实现 (Macro/Quant/Risk/CIO LangGraph)
- Phase 8: Vue3 完整前端构建 / 30日纸上交易模拟
- Phase 2: fincore/jh-factors 外部库集成
- Phase 4: S5/S15/S18/S19/S20 剩余策略
