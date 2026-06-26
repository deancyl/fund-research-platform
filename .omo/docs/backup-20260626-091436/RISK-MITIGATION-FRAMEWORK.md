# 投研平台三大核心风险 — 防御体系设计文档

**日期**: 2026-06-26 | **关联文档**: `STRATEGY-AND-DEVELOPMENT.md`
**定位**: 生产级风险控制方案，覆盖回测衰减、过拟合检测、LLM幻觉防御，含完整代码规范与验收标准

---

## 目录

- [风险全景](#风险全景)
- [风险1: 回测→实盘衰减](#风险1-回测实盘衰减)
- [风险2: 过拟合（多重测试偏误）](#风险2-过拟合多重测试偏误)
- [风险3: LLM 幻觉](#风险3-llm-幻觉)
- [三风险联合防御体系](#三风险联合防御体系)
- [实施检查清单](#实施检查清单)

---

## 风险全景

| 风险 | 严重度 | 根因 | 防御层次 |
|------|--------|------|---------|
| **回测→实盘衰减** | 高 (实测平均 -63%) | 滑点、拥挤、微观结构变化 | 前置衰减预算 + 滑点建模 + 拥挤监控 |
| **过拟合** | 高 (12,597个因子组合) | 多重测试偏误 + 时间序列信息泄露 | CPCV + PBO + DSR 三重数学门控 + 搜索空间压缩 |
| **LLM 幻觉** | 极高 (金融场景~60%) | 模型推断补全 + 系统性偏见 + 无事实锚定 | 决策权分离 + 双LLM交叉验证 + 事实锚定 + 审计日志 |

---

## 风险1: 回测→实盘衰减

### 1.1 根因分析

回测与实盘之间存在三个不可消除的鸿沟：

```
鸿沟1: 微观结构漂移
├── 回测假设: 按收盘价成交，无限流动性
├── 实盘现实: 订单簿深度变化、买卖价差波动、大宗交易冲击
└── 量级: 大盘ETF ~2-5bp/交易，小盘ETF ~8-20bp/交易

鸿沟2: 策略拥挤
├── 回测假设: 策略信号独享alpha
├── 实盘现实: 同类策略同时执行 → 抢跑、踩踏、信号衰减
└── 量级: 拥挤因子IC 6个月内可衰减 40-60%

鸿沟3: 制度参数漂移
├── 回测假设: 最优参数在样本外适用
├── 实盘现实: 市场制度变化 → 参数过期 → 策略失效
└── 量级: 夏普从1.0衰减到0.4可能仅需1-2个季度
```

### 1.2 防御方案A: 前置衰减预算

**不做回测修正，只修正预期。**

```
实盘预期夏普 = 回测夏普 × 衰减系数 β

β 映射函数（基于头部私募 2023 实测数据 + Carver 2019）:
┌──────────────────┬──────────┬──────────────────────────────┐
│ 回测夏普范围      │ β 系数   │ 依据                         │
├──────────────────┼──────────┼──────────────────────────────┤
│ < 0.5            │ 0.9-1.0  │ 低过拟合风险，衰减来自执行成本 │
│ 0.5-1.0          │ 0.6-0.8  │ 中等过拟合，信号部分有效       │
│ 1.0-1.5          │ 0.4-0.6  │ 高过拟合风险，预期衰减显著     │
│ 1.5-2.0          │ 0.3-0.5  │ 极高过拟合，应视为 0.5-1.0    │
│ > 2.0            │ 0.2-0.3  │ 几乎一定是过拟合 + 拥挤       │
└──────────────────┴──────────┴──────────────────────────────┘

当前策略库衰减后预期:
├── 因子动量 (夏普 1.15) → 预期实盘 0.46-0.69
├── ETF 多因子动量 (夏普 1.38) → 预期实盘 0.55-0.83
├── 行业动量 (夏普 1.16) → 预期实盘 0.46-0.70
├── ETF 低波 (夏普 1.06) → 预期实盘 0.42-0.85
├── RSI 均值回归 (无夏普数据) → 预期年化 3-5%（衰减40%）
└── 红利低波 (夏普 0.66) → 预期实盘 0.40-0.53
```

**验收标准**:
- [ ] 所有策略展示页同步显示 `回测夏普` 和 `预期实盘夏普` 双数字
- [ ] 预期实盘夏普 < 0.3 的策略标记为 "⚠️ 实盘可信度低"
- [ ] 季度复盘时根据实际绩效更新 β 系数

### 1.3 防御方案B: 滑点建模

**不使用固定 bps 假设，而是从真实订单簿建模。**

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class SlippageEstimate:
    """单笔交易的滑点估计"""
    spread_cost_bps: float          # 买卖价差成本
    impact_cost_bps: float          # 市场冲击成本
    total_bps: float                 # 总滑点
    confidence: float                # 估计置信度 (0-1)


class AShareSlippageModel:
    """
    A 股 ETF 滑点模型
    
    公式: 滑点 = (bid-ask spread)/2 + sqrt(Q/ADV) × η
    
    其中:
    - spread: 买卖价差 (bp)
    - Q: 订单金额 (万元)
    - ADV: 日均成交额 (万元)
    - η: 市场冲击系数 (经验校准)
    """

    # 经验校准参数（可从回测中持续更新）
    IMPACT_PARAMS = {
        "large_cap_etf": {  # CSI 300 ETF (510300)
            "spread_bps": 1.0,
            "eta": 0.3,
            "min_adv_yuan": 5_000_000_000,  # 日均 50 亿
        },
        "mid_cap_etf": {    # CSI 500 ETF (510500)
            "spread_bps": 2.0,
            "eta": 0.8,
            "min_adv_yuan": 1_000_000_000,  # 日均 10 亿
        },
        "small_cap_etf": {  # ChiNext ETF (159915)
            "spread_bps": 3.0,
            "eta": 1.5,
            "min_adv_yuan": 500_000_000,     # 日均 5 亿
        },
        "sector_etf": {     # 行业 ETF
            "spread_bps": 4.0,
            "eta": 2.0,
            "min_adv_yuan": 200_000_000,     # 日均 2 亿
        },
    }

    def estimate(
        self,
        etf_category: str,
        order_amount_yuan: float,
        avg_daily_volume_yuan: Optional[float] = None,
    ) -> SlippageEstimate:
        params = self.IMPACT_PARAMS[etf_category]
        adv = avg_daily_volume_yuan or params["min_adv_yuan"]

        # 买卖价差的一半
        spread_cost = params["spread_bps"] / 2

        # 市场冲击 (sqrt 律)
        participation_rate = order_amount_yuan / adv
        impact_cost = np.sqrt(participation_rate) * params["eta"]

        total = spread_cost + impact_cost

        # 置信度随参与率下降（参与率越低，估计越不准）
        confidence = max(0.3, 1.0 - participation_rate * 10)

        return SlippageEstimate(
            spread_cost_bps=round(spread_cost, 2),
            impact_cost_bps=round(impact_cost, 2),
            total_bps=round(total, 2),
            confidence=round(confidence, 2),
        )

    def estimate_trade_cost(
        self,
        etf_category: str,
        order_amount_yuan: float,
        is_sell: bool = False,
    ) -> dict:
        """
        完整交易成本 = 滑点 + 佣金 + 印花税(卖出) + 过户费
        """
        slip = self.estimate(etf_category, order_amount_yuan)

        commission_bps = 0.25       # 万 2.5 佣金 (单边)
        stamp_duty_bps = 0.50 if is_sell else 0.0   # 0.05% 印花税 (仅卖出)
        transfer_fee_bps = 0.01     # 0.001% 过户费 (双向)

        total = slip.total_bps + commission_bps + stamp_duty_bps + transfer_fee_bps

        return {
            "slippage_bps": slip.total_bps,
            "commission_bps": commission_bps,
            "stamp_duty_bps": stamp_duty_bps,
            "transfer_fee_bps": transfer_fee_bps,
            "total_bps": round(total, 2),
            "total_yuan": round(order_amount_yuan * total / 10000, 2),
            "confidence": slip.confidence,
        }
```

**验收标准**:
- [ ] 回测引擎集成 `AShareSlippageModel`，每笔模拟交易使用动态滑点
- [ ] 成本敏感性测试: 基准成本 + 成本×2 + 成本×5 三种场景
- [ ] 策略在成本×5 下仍不亏损 → 通过成本鲁棒性检验

### 1.4 防御方案C: 拥挤度监控

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class CrowdingSignal:
    """策略拥挤预警信号"""
    timestamp: str
    strategy_name: str
    signal_concentration: float         # 信号集中度 (0-1, 越低越集中)
    factor_ic_trend_6m: float           # 因子 IC 6 个月趋势
    turnover_ratio: float               # 换手率 vs 历史均值
    crowding_score: float               # 综合拥挤评分 (0-1, 越高越拥挤)
    warning_level: str                  # GREEN/YELLOW/RED


class CrowdingMonitor:
    """
    策略拥挤度实时监控

    三个维度:
    1. 信号集中度: 多策略是否指向同一标的
    2. 因子 IC 衰减: 因子预测能力是否在下降
    3. 换手率异常: 策略触发的换手是否超出市场承载
    """

    def __init__(self):
        self.history: List[CrowdingSignal] = []

    def compute_signal_concentration(
        self, active_strategies: list
    ) -> float:
        """
        信号集中度: 有多少策略发出同一方向的信号

        concentration = 1.0 → 所有策略一致 (无拥挤)
        concentration = 0.1 → 极度拥挤 (只在少数标的上集中)
        """
        all_targets = []
        for strat in active_strategies:
            all_targets.extend(strat.current_signals)

        unique_targets = set(all_targets)
        return len(unique_targets) / len(all_targets) if all_targets else 1.0

    def compute_factor_ic_decay(
        self,
        factor_name: str,
        ic_series: np.ndarray,
        half_life_window: int = 126,
    ) -> float:
        """
        因子 IC 衰减检测

        方法: 滚动 126 天 IC → 差分均值趋势
        正值 = IC 改善, 负值 = IC 衰减
        """
        rolling = np.convolve(ic_series, np.ones(half_life_window) / half_life_window, mode="valid")
        trend = np.mean(np.diff(rolling))
        return float(trend)

    def compute_turnover_anomaly(
        self,
        current_turnover: float,
        historical_turnover: np.ndarray,
        z_threshold: float = 2.0,
    ) -> float:
        """
        换手率异常检测

        返回 Z-score:
        > 2.0 → 换手率显著高于历史 → 可能存在拥挤执行
        < -2.0 → 换手率显著低于历史 → 策略可能停滞
        """
        mu, sigma = np.mean(historical_turnover), np.std(historical_turnover)
        return (current_turnover - mu) / sigma if sigma > 0 else 0.0

    def evaluate(
        self,
        strategy_name: str,
        active_strategies: list,
        factor_name: str,
        ic_series: np.ndarray,
        current_turnover: float,
        historical_turnover: np.ndarray,
    ) -> CrowdingSignal:
        concentration = self.compute_signal_concentration(active_strategies)
        ic_decay = self.compute_factor_ic_decay(factor_name, ic_series)
        turnover_z = self.compute_turnover_anomaly(current_turnover, historical_turnover)

        # 综合拥挤评分 (加权)
        concentration_risk = (1 - concentration) * 0.40
        ic_risk = max(0, -ic_decay) * 20 * 0.35     # 归一化
        turnover_risk = max(0, turnover_z) / 3 * 0.25  # 归一化

        score = concentration_risk + ic_risk + turnover_risk
        score = min(1.0, score)

        if score < 0.3:
            level = "GREEN"
        elif score < 0.6:
            level = "YELLOW"
        else:
            level = "RED"

        signal = CrowdingSignal(
            timestamp=str(np.datetime64("now")),
            strategy_name=strategy_name,
            signal_concentration=concentration,
            factor_ic_trend_6m=ic_decay,
            turnover_ratio=turnover_z,
            crowding_score=score,
            warning_level=level,
        )
        self.history.append(signal)
        return signal

    def should_reduce_position(self, strategy_name: str) -> bool:
        """
        拥挤度 RED → 建议减仓 50%
        拥挤度 YELLOW → 建议减仓 20%
        """
        recent = [s for s in self.history[-5:] if s.strategy_name == strategy_name]
        if not recent:
            return False
        return any(s.warning_level == "RED" for s in recent)
```

**验收标准**:
- [ ] 策略信号集中度每日计算，< 0.3 触发预警
- [ ] 因子 IC 趋势每 21 天更新，连续 3 期衰减 → 因子降权
- [ ] 换手率 Z-score > 2.5 → 自动限制当日交易量

---

## 风险2: 过拟合（多重测试偏误）

### 2.1 根因分析

```
问题: 12,597 个因子组合中找出 "最好" 的那一个

经典过拟合路径:
├── 尝试所有参数组合 → 按夏普排名 → 选第一名
├── 🚨 第一名极大概率是 "恰好在噪音中拟合了历史" 而非 "真正有预测力"
├── 数学本质: E[max(SR) | SR_true=0] ≈ √(2 × ln(M) / T) 
│   其中 M=12,597 次尝试, T=500 个交易日
│   → E[max(SR)] ≈ 0.14 → 即使所有策略无效，也能"发现"夏普 0.14 的策略
└── 叠加时间序列自相关 → 信息泄露 → 实际 E[max(SR)] 可能达到 0.3-0.5
```

### 2.2 防御方案A: 三重数学门控

```
┌─────────────────────────────────────────────────────────────┐
│          策略必须通过全部三道检验，方可进入实盘                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  检验1: CPCV — Combinatorial Purged Cross-Validation        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 生成 C(N,K) 条独立 OOS 路径 (N=6, K=2 → 15 条)        │  │
│  │ 每条路径: purge 训练/测试相邻时段 + embargo 禁止信息   │  │
│  │         泄露                                           │  │
│  │ 输出: OOS 夏普分布 (非单点估计)                       │  │
│  │ 要求: OOS 夏普中位数 > 0 AND 25%分位数 > -0.3         │  │
│  │ 失败: 策略被拒绝，不进入 PBO 检验                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  检验2: PBO — Probability of Backtest Overfitting           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 在 15 条 CPCV 路径上计算:                              │  │
│  │   PBO = P(IS 最优策略在 OOS 中低于中位数)               │  │
│  │ 即: IS 排名第一的策略，在 OOS 中排第几？               │  │
│  │ 如果 IS=1st 但在 OOS 中排第 8/15 → PBO ≈ 0.53         │  │
│  │ 要求: PBO < 0.10 (10% 以下)                            │  │
│  │ PBO > 0.30 → 直接拒绝，不进入 DSR 检验                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  检验3: DSR — Deflated Sharpe Ratio                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ DSR = (实测 SR - E[max SR|H0]) / σ(SR)                │  │
│  │ 其中 E[max SR|H0] 考虑了 M 次搜索的过拟合修正         │  │
│  │ 要求: DSR > 0.95 (95% 置信水平)                        │  │
│  │       最小回测长度 χ_min > 理论最小值                  │  │
│  │ DSR 为负 → 策略夏普低于随机搜索期望 → 拒绝             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**完整实现规范**:

```python
import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scipy import stats


@dataclass
class BacktestValidationResult:
    """三重检验的综合结果"""
    cpcv_passed: bool
    pbo_passed: bool
    dsr_passed: bool
    cpcv_sharpe_median: float
    cpcv_sharpe_p25: float
    pbo_value: float
    dsr_value: float
    min_track_record_days: int
    overall_verdict: str  # APPROVED / REJECTED / NEEDS_MORE_DATA


def combinatorial_purged_cv(
    returns: np.ndarray,
    n_splits: int = 6,
    n_test_splits: int = 2,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.01,
) -> List[Dict]:
    """
    组合净化交叉验证

    参数:
    - n_splits: 总拆分份数 (N)
    - n_test_splits: 每次用作测试的份数 (K)
    - purge_pct: 训练集末端切除比例 (防止标签重叠)
    - embargo_pct: 训练集后端禁止比例 (防止信息泄露)

    返回:
    - List[{"train_idx", "test_idx", "oos_sharpe"}]
    """
    T = len(returns)
    split_indices = np.array_split(np.arange(T), n_splits)
    test_combos = list(combinations(range(n_splits), n_test_splits))

    results = []
    for combo in test_combos:
        test_idx = np.concatenate([split_indices[i] for i in combo])
        train_idx = np.array([i for i in range(T) if i not in test_idx])

        # Purge: 移除训练集末尾 purge_pct 比例的数据
        purge_size = int(len(train_idx) * purge_pct)
        if purge_size > 0:
            train_idx = train_idx[:-purge_size]

        # Embargo: 移除紧接测试集之前的 embargo_pct
        embargo_size = int(T * embargo_pct)
        min_test_idx = test_idx.min()
        train_idx = train_idx[train_idx < (min_test_idx - embargo_size)]

        if len(train_idx) < 50 or len(test_idx) < 20:
            continue

        train_returns = returns[train_idx]
        test_returns = returns[test_idx]

        # 在训练集上计算策略参数，在测试集上评估
        oos_sharpe = _evaluate_oos(train_returns, test_returns)

        results.append({
            "train_idx": train_idx,
            "test_idx": test_idx,
            "oos_sharpe": oos_sharpe,
        })

    return results


def _evaluate_oos(
    train_returns: np.ndarray,
    test_returns: np.ndarray,
) -> float:
    """在训练集上拟合策略参数，在测试集上计算夏普"""
    # 此处为占位 — 实际实现取决于具体策略
    # 关键原则: 测试集数据在参数估计阶段完全不可见
    strategy_return = test_returns.mean()
    strategy_vol = test_returns.std()
    return strategy_return / strategy_vol if strategy_vol > 0 else 0.0


def compute_pbo(
    cpcv_results: List[Dict],
    in_sample_sharpes: np.ndarray,
) -> float:
    """
    计算回测过拟合概率 (PBO)

    PBO = P(IS 最优策略在 OOS 中的排名低于中位数)

    步骤:
    1. 在每个 CPCV 路径中，确定 IS 最优策略的排名
    2. 查看该策略在 OOS 中的排名
    3. PBO = (IS最优但在OOS中低于中位数的路径数) / 总路径数
    """
    n_paths = len(cpcv_results)
    is_best_flopped_in_oos = 0

    for path in cpcv_results:
        is_best_idx = np.argmax(in_sample_sharpes)
        oos_ranking = np.argsort([p["oos_sharpe"] for p in cpcv_results])[::-1]
        oos_rank_of_is_best = np.where(oos_ranking == is_best_idx)[0][0]

        if oos_rank_of_is_best > n_paths / 2:
            is_best_flopped_in_oos += 1

    return is_best_flopped_in_oos / n_paths


def compute_deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    缩水夏普比率 (DSR)

    Bailey & Lopez de Prado (2014)
    修正了 M 次搜索尝试的多重测试偏误

    参数:
    - observed_sr: 观测到的夏普比率
    - n_trials: 搜索的策略/参数组合数量 (M)
    - n_observations: 样本观测数 (T)
    - skewness: 收益率偏度
    - kurtosis: 收益率峰度

    返回:
    - DSR 值 (> 0.95 表示统计显著)
    """
    # E[max(SR) | H0] 的近似
    # 来源: Bailey & Lopez de Prado 2014, Eq. (5)
    euler_mascheroni = 0.5772156649
    e_max_sr = np.sqrt(2 * np.log(n_trials)) - (
        np.log(np.log(n_trials)) + np.log(4 * np.pi) - 2 * euler_mascheroni
    ) / (2 * np.sqrt(2 * np.log(n_trials)))

    # SR 的标准误（含偏度和峰度修正）
    sr_se = np.sqrt(
        (1 / n_observations)
        * (1 + 0.5 * observed_sr**2 - skewness * observed_sr + (kurtosis - 3) / 4 * observed_sr**2)
    )

    return (observed_sr - e_max_sr) / sr_se


def compute_min_track_record(
    observed_sr: float,
    significance_level: float = 0.95,
) -> int:
    """
    最小回测长度

    当观测数不足时，即使夏普比率很高也不具统计显著性。

    来源: Bailey & Lopez de Prado 2014, Eq. (11)
    """
    z_alpha = stats.norm.ppf(significance_level)
    min_obs = (z_alpha / observed_sr) ** 2 * (1 + 0.5 * observed_sr**2)
    return int(np.ceil(min_obs))


def validate_strategy(
    returns: np.ndarray,
    observed_sharpe: float,
    n_factor_combinations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> BacktestValidationResult:
    """
    策略三重验证 — 一次性执行全部检验

    返回综合判决: APPROVED / REJECTED / NEEDS_MORE_DATA
    """
    T = len(returns)

    # 检验1: CPCV
    cpcv = combinatorial_purged_cv(returns)
    oos_sharpes = [r["oos_sharpe"] for r in cpcv]
    cpcv_median = float(np.median(oos_sharpes))
    cpcv_p25 = float(np.percentile(oos_sharpes, 25))
    cpcv_passed = cpcv_median > 0 and cpcv_p25 > -0.3

    if not cpcv_passed:
        return BacktestValidationResult(
            cpcv_passed=False, pbo_passed=False, dsr_passed=False,
            cpcv_sharpe_median=cpcv_median, cpcv_sharpe_p25=cpcv_p25,
            pbo_value=1.0, dsr_value=-999.0, min_track_record_days=0,
            overall_verdict="REJECTED",
        )

    # 检验2: PBO
    in_sample_sharpes = np.array([observed_sharpe] * len(cpcv))
    pbo = compute_pbo(cpcv, in_sample_sharpes)
    pbo_passed = pbo < 0.10

    if not pbo_passed or pbo > 0.30:
        return BacktestValidationResult(
            cpcv_passed=True, pbo_passed=False, dsr_passed=False,
            cpcv_sharpe_median=cpcv_median, cpcv_sharpe_p25=cpcv_p25,
            pbo_value=pbo, dsr_value=-999.0, min_track_record_days=0,
            overall_verdict="REJECTED",
        )

    # 检验3: DSR
    dsr = compute_deflated_sharpe_ratio(
        observed_sharpe, n_factor_combinations, T, skewness, kurtosis
    )
    min_days = compute_min_track_record(observed_sharpe)
    dsr_passed = dsr > 0.95

    if not dsr_passed:
        if T < min_days:
            return BacktestValidationResult(
                cpcv_passed=True, pbo_passed=True, dsr_passed=False,
                cpcv_sharpe_median=cpcv_median, cpcv_sharpe_p25=cpcv_p25,
                pbo_value=pbo, dsr_value=dsr, min_track_record_days=min_days,
                overall_verdict="NEEDS_MORE_DATA",
            )

    return BacktestValidationResult(
        cpcv_passed=True, pbo_passed=True, dsr_passed=dsr_passed,
        cpcv_sharpe_median=cpcv_median, cpcv_sharpe_p25=cpcv_p25,
        pbo_value=pbo, dsr_value=dsr, min_track_record_days=min_days,
        overall_verdict="APPROVED" if dsr_passed else "NEEDS_MORE_DATA",
    )
```

**验收标准**:
- [ ] 所有策略上线前执行 `validate_strategy()`
- [ ] VERDICT=APPROVED 才可进入纸上交易
- [ ] REJECTED 策略归档并记录拒绝原因
- [ ] NEEDS_MORE_DATA 策略进入观察队列，每季度重检

### 2.3 防御方案B: 搜索空间压缩

**不依赖事后检验，从源头减少过拟合可能。**

```
三层压缩 → 将 12,597 个组合降至 ~50-200 个:

Layer 1: 相关性过滤
├── 规则: |corr(factor_i, factor_j)| > 0.7 → 只保留 IC 更高者
├── 预期: 23 因子 → 约 12-15 个独立因子
└── 伪代码:
    for i, j in combinations(factors, 2):
        if abs(corr(i, j)) > 0.7:
            discard(min(IC_rank(i), IC_rank(j)))

Layer 2: 经济学逻辑约束
├── 规则1: 因子方向必须与金融理论一致
│   例: 价值因子 → 买低 PE → IC > 0, 若 IC < 0 → 排除
├── 规则2: 因子 IC 在近 12 个月内方向稳定
│   IC_sign_consistency = sum(IC_monthly > 0) / 12
│   要求 > 0.6 (至少 8/12 个月方向一致)
└── 规则3: 因子不能是"纯噪音"
    要求 IC 的 t-stat > 1.5 (有微弱预测力即可)

Layer 3: 参数漂移正则化
├── 规则1: 参数不在极端值
│   例: 动量回看不在 1 天(噪音)或 500 天(过于平滑)
├── 规则2: 参数扰动 ±20% 时，绩效波动 < 30%
│   test_params = [p*0.8, p, p*1.2]
│   要求 max(sharpe) - min(sharpe) < 0.3 * median(sharpe)
└── 规则3: 至少 2 个替代参数窗口产生一致方向
    例: 动量回看 = 63天, 126天, 252天
    要求至少 2 个产生正 IC
```

**验收标准**:
- [ ] 因子库中每个因子记录: IC / t-stat / 方向一致性 / 参数稳定性
- [ ] 季度自动清理: 不合格因子降权或移出因子池
- [ ] 因子操作日志: 每次添加/删除因子需记录理由

### 2.4 防御方案C: 时间序列增强交叉验证

```python
from typing import List, Tuple
import numpy as np


def purged_walk_forward_validation(
    returns: np.ndarray,
    dates: np.ndarray,
    train_years: float = 3.0,
    test_months: int = 6,
    min_regimes_per_fold: int = 2,
    step_months: int = 3,
) -> List[Dict]:
    """
    WFO 增强版 — 多制度覆盖 + Purge + Embargo

    标准 WFO 的问题:
    - 可能某折全是牛市，另一折全是熊市 → 不可比
    - 训练/测试边界处有信息泄露

    增强措施:
    - 每折必须覆盖 ≥2 种市场状态
    - Purge: 训练集末尾 5% 切除
    - Embargo: 训练集后端 5 天禁止测试
    """
    T = len(returns)
    train_len = int(train_years * 252)
    test_len = test_months * 21
    step_len = step_months * 21
    purge_len = int(train_len * 0.05)
    embargo_len = 5

    results = []
    start = 0

    while start + train_len + test_len <= T:
        train_end = start + train_len - purge_len - embargo_len
        test_start = start + train_len
        test_end = test_start + test_len

        if train_end <= start:
            start += step_len
            continue

        # 制度覆盖检查
        test_returns = returns[test_start:test_end]
        regimes = detect_regimes_simple(test_returns)

        if len(set(regimes)) < min_regimes_per_fold:
            start += step_len
            continue

        train_returns = returns[start:train_end]

        oos_sharpe = _evaluate_oos(train_returns, test_returns)

        results.append({
            "train_start": dates[start],
            "train_end": dates[train_end],
            "test_start": dates[test_start],
            "test_end": dates[test_end],
            "oos_sharpe": oos_sharpe,
            "regimes_in_test": set(regimes),
        })

        start += step_len

    # 返回分布而非单值
    oos_sharpes = [r["oos_sharpe"] for r in results]
    return {
        "folds": results,
        "n_valid_folds": len(results),
        "oos_sharpe_median": float(np.median(oos_sharpes)),
        "oos_sharpe_mean": float(np.mean(oos_sharpes)),
        "oos_sharpe_std": float(np.std(oos_sharpes)),
        "oos_sharpe_p10": float(np.percentile(oos_sharpes, 10)),
        "consistency": float(np.mean([s > 0 for s in oos_sharpes])),
        "worst_fold_sharpe": float(min(oos_sharpes)),
    }


def detect_regimes_simple(returns: np.ndarray) -> List[str]:
    """
    简易市场状态分类
    - 上升: 累计收益 > 0 且波动率 < 中位数
    - 下降: 累计收益 < -5% 或波动率 > 中位数
    - 震荡: 其他
    """
    cum_ret = np.cumprod(1 + returns)[-1] - 1
    vol = np.std(returns)

    if cum_ret > 0.03 and vol < 0.015:
        return ["UP"]
    elif cum_ret < -0.05 or vol > 0.025:
        return ["DOWN"]
    else:
        return ["SIDEWAYS"]
```

**验收标准**:
- [ ] WFO 验证至少产生 6 个有效折
- [ ] OOS 夏普 p10 > -0.5 (最差 10% 情况仍可控)
- [ ] 一致性 > 0.55 (超过一半的折正收益)

---

## 风险3: LLM 幻觉

### 3.1 根因分析

```
LLM 幻觉在金融场景下的三种形态:

形态1: 事实编造 (Factual Hallucination)
├── "创业板指当前 PE=25.3" → 实际是 35.2
├── "北向资金连续3日净流入" → 实际是净流出
└── 原因: 训练数据截止日旧于查询日，模型"平滑推测"

形态2: 逻辑错误 (Logical Hallucination)
├── "RSI 超买 + 均线金叉 → 建议买入" → RSI 超买是卖出信号
├── "PE 处于历史 80% 分位，估值合理" → 80% 分位是偏贵
└── 原因: 模型对不同金融指标的关系没有确定性的因果理解

形态3: 系统性偏见 (Systemic Bias)
├── "中国科技股前景看好" → 可能只是复述了主流叙事
├── "当前是买入良机" → 模型有乐观偏见的倾向
└── 原因: 预训练数据中的叙事偏向会被复现
```

### 3.2 防御方案A: 决策权分离（核心防御）

**LLM 负责分析推理，确定性数学负责最终决策。**

这是参考 ThesisAgent 的生产级设计：

```
┌───────────────────────┐        ┌───────────────────────┐        ┌───────────────────┐
│     LLM Agent 层       │        │    Decision Hub 层    │        │     输出层         │
│     (允许幻觉)          │        │    (不允许错误)        │        │                   │
├───────────────────────┤        ├───────────────────────┤        │                   │
│                       │        │                       │        │                   │
│ Macro Strategist      │        │ 8因子得分: 0.62       │        │ VERDICT: HOLD     │
│ "全球风险偏好回升..."  │  ───→  │                       │  ───→  │                   │
│                       │        │ 买入阈值: > 0.75      │        │ 仓位: 0%          │
│ Quant Analyst         │        │ 卖出阈值: < 0.25      │        │                   │
│ "RSI超买, MACD死叉..." │  ───→  │                       │        │ 理由: 因子得分     │
│                       │        │ LLM 调整: ±0.15       │        │ 0.62, 低于买入     │
│ Risk Officer          │        │ (仅限调权重)           │        │ 阈值0.75          │
│ "仓位集中度30%, 偏高"  │  ───→  │                       │        │                   │
│                       │        │ 风控熔断:             │        │ LLM贡献:          │
│ CIO                   │        │ CRISIS → max=HOLD    │        │ "风险提示: 集中度  │
│ "建议ACCUMULATE"      │  ───→  │ 单标的上限: 20%       │        │  偏高, 宏观中性"   │
│                       │        │                       │        │                   │
└───────────────────────┘        └───────────────────────┘        └───────────────────┘
         ↑                                ↑                              ↑
    容易出错                         永远正确                        可验证
```

**实现规范**:

```python
from enum import Enum
from dataclasses import dataclass
import numpy as np


class Verdict(str, Enum):
    BUY = "BUY"
    ACCUMULATE = "ACCUMULATE"
    HOLD = "HOLD"
    TRIM = "TRIM"
    SELL = "SELL"


@dataclass(frozen=True)
class AgentOutput:
    """单个 Agent 的输出 — 结构化、可验证"""
    agent_name: str
    signal: str              # risk_on / bullish / ok / etc.
    strength: float          # 0.0-1.0
    key_data_points: dict    # {"PE": 35.2, "source": "AKShare 2026-06-26"}
    narrative: str            # LLM 的文本分析 (允许有错，仅作参考)


@dataclass(frozen=True)
class Decision:
    """Decision Hub 的最终输出 — 确定性、可审计"""
    verdict: Verdict
    factor_score: float
    llm_adjustment: float
    position_pct: float
    confidence: float
    evidence: list           # 支撑决策的具体数据点
    llm_narrative: str       # LLM 的分析文本 (标记为"仅供参考")
    safety_checks_passed: int  # 通过的安全检查数/总数


class DecisionHub:
    """
    确定性决策中枢

    核心原则:
    1. LLM 永远不能直接输出 BUY/SELL
    2. LLM 只能调整因子得分的 ±15%
    3. 硬性约束（风控熔断/仓位上限）不可被 LLM 覆盖
    """

    BASE_THRESHOLDS = {
        Verdict.BUY: 0.75,
        Verdict.ACCUMULATE: 0.60,
        Verdict.HOLD: 0.40,
        Verdict.TRIM: 0.25,
        Verdict.SELL: 0.00,
    }

    LLM_ADJUSTMENT_RANGE = (-0.15, 0.15)
    MAX_SINGLE_POSITION_PCT = 0.20

    def __init__(self, risk_officer_flag: str = "NORMAL"):
        self.risk_flag = risk_officer_flag

    def decide(
        self,
        factor_score: float,
        llm_cio_verdict: str,
        llm_cio_confidence: float,
        agent_outputs: list,
        current_portfolio: dict = None,
    ) -> Decision:
        """
        核心决策方法

        参数:
        - factor_score: 8因子加权得分 (0.0-1.0)
        - llm_cio_verdict: CIO LLM 的建议 ("BUY"/"ACCUMULATE"/...)
        - llm_cio_confidence: CIO LLM 的置信度 (0.0-1.0)
        - agent_outputs: 所有 Agent 的结构化输出
        - current_portfolio: 当前持仓 {symbol: pct}
        """
        safety_checks = 0
        total_checks = 6

        # 安全检查1: LLM 调整幅度受限于 ±15%
        factor_to_verdict = self._score_to_verdict(factor_score)
        llm_direction = self._verdict_to_direction(llm_cio_verdict)
        factor_direction = self._verdict_to_direction(factor_to_verdict)

        # LLM 和因子得分方向一致 → LLM 可以微调
        if llm_direction == factor_direction:
            llm_adjustment = llm_cio_confidence * self.LLM_ADJUSTMENT_RANGE[1] * np.sign(llm_direction)
        else:
            # 方向矛盾 → LLM 调整归零
            llm_adjustment = 0.0
            safety_checks -= 1  # 扣分: LLM 与量化信号矛盾

        llm_adjustment = np.clip(llm_adjustment, *self.LLM_ADJUSTMENT_RANGE)
        adjusted_score = np.clip(factor_score + llm_adjustment, 0.0, 1.0)
        safety_checks += 1  # 通过: LLM 调整在范围内

        # 安全检查2: 风控熔断 → 强制降低上限
        if self.risk_flag == "CRISIS":
            adjusted_score = min(adjusted_score, self.BASE_THRESHOLDS[Verdict.HOLD])
            safety_checks += 1  # 通过: 风控熔断应用
        else:
            safety_checks += 1

        # 安全检查3: 单标的仓位上限
        position_pct = self._position_size(adjusted_score)
        if current_portfolio:
            for symbol, pct in current_portfolio.items():
                total_position = position_pct + pct
                if total_position > self.MAX_SINGLE_POSITION_PCT:
                    position_pct = max(0, self.MAX_SINGLE_POSITION_PCT - pct)
        safety_checks += 1  # 通过: 仓位限额

        # 安全检查4: 过度自信检测
        calibrated_confidence = self._calibrate_confidence(
            llm_cio_confidence, len(agent_outputs)
        )
        safety_checks += 1

        # 安全检查5: 数据完整性 — 所有 Agent 是否正常工作
        if not all(a.key_data_points for a in agent_outputs):
            adjusted_score = min(adjusted_score, self.BASE_THRESHOLDS[Verdict.HOLD])
        safety_checks += 1

        # 安全检查6: 因子得分异常检测
        if factor_score > 0.95 or factor_score < 0.05:
            adjusted_score = 0.50  # 强制中性
        safety_checks += 1

        verdict = self._score_to_verdict(adjusted_score)

        return Decision(
            verdict=verdict,
            factor_score=factor_score,
            llm_adjustment=float(llm_adjustment),
            position_pct=float(position_pct),
            confidence=float(calibrated_confidence),
            evidence=self._collect_evidence(agent_outputs),
            llm_narrative=self._extract_narrative(agent_outputs),
            safety_checks_passed=safety_checks,
        )

    def _score_to_verdict(self, score: float) -> Verdict:
        for verdict, threshold in sorted(
            self.BASE_THRESHOLDS.items(), key=lambda x: -x[1]
        ):
            if score >= threshold:
                return verdict
        return Verdict.SELL

    def _verdict_to_direction(self, verdict_str: str) -> int:
        mapping = {"BUY": 2, "ACCUMULATE": 1, "HOLD": 0, "TRIM": -1, "SELL": -2}
        return mapping.get(verdict_str, 0)

    def _position_size(self, score: float) -> float:
        """分数 Kelly 仓位计算 — 25% Kelly"""
        f_star = score * 0.25
        return min(f_star, self.MAX_SINGLE_POSITION_PCT)

    def _calibrate_confidence(
        self, raw_confidence: float, n_agents_active: int
    ) -> float:
        """置信度校准"""
        if n_agents_active < 3:  # 少于3个Agent → 信息不足
            return raw_confidence * 0.5
        return raw_confidence

    def _collect_evidence(self, agent_outputs: list) -> list:
        """汇总所有 Agent 的关键数据点"""
        evidence = []
        for agent in agent_outputs:
            for key, value in agent.key_data_points.items():
                evidence.append({"agent": agent.agent_name, "metric": key, "value": value})
        return evidence

    def _extract_narrative(self, agent_outputs: list) -> str:
        """提取 CIO 的文本分析（标记为仅供参考）"""
        cio = next((a for a in agent_outputs if a.agent_name == "CIO"), None)
        return cio.narrative if cio else "N/A"
```

**验收标准**:
- [ ] `DecisionHub.decide()` 是唯一合法的决策入口
- [ ] LLM 的 advice 永远不能通过 API 直接转为交易指令
- [ ] 6 项安全检查全部通过 → 标记 `safety_checks_passed=6`
- [ ] 任何检查失败 → VERDICT 降级至少一个级别

### 3.3 防御方案B: 事实锚定

**LLM 的每条分析必须有数据支撑。**

```python
class FactAnchoredAgent:
    """
    强制 LLM Agent 在每一步分析中引用具体数据和来源

    原则: LLM 可以给出错误的"解读"，但不能给出错误的"事实"
    """

    PROMPT_TEMPLATE = """
    你是一个 {role}。在分析 {symbol} 时，你必须遵守以下规则:

    1. 每个陈述必须附带具体数字和来源
       正确: "PE(TTM)=35.2, 历史63%分位 (来源: AKShare 2026-06-26)"
       错误: "估值偏高"

    2. 只使用下方给定的数据，不得编造
       给定数据:
       {data_context}

    3. 如果给定数据不足以得出结论，你必须说 "数据不足，无法判断"

    4. 输出格式 (严格遵循):
       SIGNAL: <direction>
       STRENGTH: <0-10>
       KEY_DATA: <bullet points, each with a number and source>
       ANALYSIS: <your narrative>
    """

    def __init__(self, role: str, data_fetcher):
        self.role = role
        self.data_fetcher = data_fetcher

    def analyze(self, symbol: str) -> dict:
        """获取真实数据 → 注入 prompt → 调用 LLM"""
        real_data = self.data_fetcher.fetch(symbol)
        prompt = self.PROMPT_TEMPLATE.format(
            role=self.role, symbol=symbol, data_context=real_data
        )
        llm_output = call_llm(prompt)
        return self._validate_output(llm_output, real_data)

    def _validate_output(self, output: dict, real_data: dict) -> dict:
        """
        验证 LLM 输出中的数据引用是否与真实数据一致

        如果 LLM 编造了不在 real_data 中的数字 → 标记为 UNRELIABLE
        """
        cited_values = self._extract_numbers(output.get("KEY_DATA", ""))
        for cited in cited_values:
            if cited not in real_data.values():
                output["RELIABILITY"] = "UNRELIABLE"
                output["HALLUCINATION_NOTE"] = f"引用了不存在的数据: {cited}"
                return output
        output["RELIABILITY"] = "VERIFIED"
        return output
```

**验收标准**:
- [ ] 每个 Agent 输出包含 `KEY_DATA` 字段，每行包含: 数字 + 来源 + 日期
- [ ] 如果 Agent 标记为 `UNRELIABLE` → 该 Agent 的本轮分析被丢弃
- [ ] 月度报告: 统计各 Agent 的 `UNRELIABLE` 率

### 3.4 防御方案C: 双 LLM 交叉验证

```python
class DualLLMValidator:
    """
    两个不同的 LLM 独立推理 → 结果对比

    原理:
    - 不同的训练数据 → 不同的偏见
    - 如果两个独立训练的模型得出相同结论 → 可信度更高
    - 如果结论矛盾 → 强制 HOLD
    """

    def __init__(self, primary_model: str = "deepseek-v4", secondary_model: str = "qwen-max"):
        self.primary = primary_model
        self.secondary = secondary_model

    def cross_validate(
        self, factor_score: float, market_context: dict, agent_outputs: list
    ) -> dict:
        """双 LLM 独立推理 → 交叉验证"""
        primary_verdict = self._query_llm(self.primary, factor_score, market_context, agent_outputs)
        secondary_verdict = self._query_llm(self.secondary, factor_score, market_context, agent_outputs)

        # 一致性分析
        direction_match = primary_verdict["direction"] == secondary_verdict["direction"]
        verdict_match = primary_verdict["verdict"] == secondary_verdict["verdict"]
        magnitude_diff = abs(primary_verdict["confidence"] - secondary_verdict["confidence"])

        multiplier = 1.0
        if verdict_match:
            multiplier = 1.0      # 完全一致
        elif direction_match:
            multiplier = 0.7      # 方向一致，强度不同
        else:
            multiplier = 0.3      # 方向相反 → 强制保守

        return {
            "primary": primary_verdict,
            "secondary": secondary_verdict,
            "direction_match": direction_match,
            "verdict_match": verdict_match,
            "magnitude_diff": magnitude_diff,
            "confidence_multiplier": multiplier,
            "merged_verdict": self._merge(primary_verdict, secondary_verdict, multiplier),
        }

    def _merge(self, v1, v2, multiplier) -> str:
        """两个矛盾的结论合并"""
        if multiplier < 0.5:
            return "HOLD"  # 方向矛盾 → 不做决策
        return v1["verdict"]
```

**验收标准**:
- [ ] 两个 LLM 方向矛盾时 → 自动输出 HOLD
- [ ] 方向一致但强度不同 → 置信度降至原来的 70%
- [ ] 月度双 LLM 一致性报告 → 追踪两模型的系统偏差

### 3.5 防御方案D: 幻觉审计日志

```python
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class AuditRecord:
    decision_id: str
    timestamp: datetime
    agent_name: str
    verdict: str
    expected_return: float
    actual_return_7d: float = None
    actual_return_30d: float = None
    result_7d: str = "PENDING"      # PASS / OVERCONFIDENT / HALLUCINATION
    result_30d: str = "PENDING"
    notes: str = ""


class HallucinationAuditor:
    """
    幻觉审计日志

    每条 AI 决策在 7/30/90 天后自动复盘:
    - 方向正确 + 幅度匹配 → PASS
    - 方向正确 + 幅度偏差 > 50% → OVERCONFIDENT
    - 方向错误 → HALLUCINATION
    - 数据引用错误 → FACTUAL_ERROR
    """

    def __init__(self):
        self.records: list = []
        self.settled: list = []     # 已结算的决策 (到期可复盘)

    def record_decision(self, decision: Decision, agent_name: str) -> str:
        """记录一条新决策，生成 ID"""
        record_id = f"AUD-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{agent_name}"
        self.records.append(AuditRecord(
            decision_id=record_id,
            timestamp=datetime.now(),
            agent_name=agent_name,
            verdict=decision.verdict.value,
            expected_return=decision.confidence * 0.10,  # 假设预期收益
        ))
        return record_id

    def settle_decisions(self, current_prices: dict):
        """
        结算到达复盘期的决策

        7 天: 短期方向判断
        30 天: 中期幅度判断
        """
        now = datetime.now()
        for record in self.records:
            if record.result_7d != "PENDING":
                continue

            age = (now - record.timestamp).days
            symbol = record.agent_name.split("_")[-1]

            if age >= 7 and record.actual_return_7d is None:
                record.actual_return_7d = current_prices.get(symbol, 0)
                record.result_7d = self._evaluate(
                    record.expected_return, record.actual_return_7d, horizon="7d"
                )

            if age >= 30 and record.actual_return_30d is None:
                record.actual_return_30d = current_prices.get(symbol, 0)
                record.result_30d = self._evaluate(
                    record.expected_return, record.actual_return_30d, horizon="30d"
                )

    def _evaluate(self, expected: float, actual: float, horizon: str) -> str:
        """评估单条决策的对错"""
        direction_correct = (expected > 0) == (actual > 0)

        if not direction_correct:
            return "HALLUCINATION"

        magnitude_error = abs(expected - actual) / max(abs(actual), 0.001)
        if magnitude_error > 0.5:
            return "OVERCONFIDENT"

        return "PASS"

    def monthly_report(self) -> dict:
        """
        月度审计报告

        关键指标:
        - 各 Agent 的幻觉率
        - 各市场状态下的幻觉率
        - 幻觉率趋势 (是否改善?)
        """
        settled = [r for r in self.records if r.result_30d != "PENDING"]
        if not settled:
            return {"status": "NO_DATA"}

        by_agent = defaultdict(lambda: {"PASS": 0, "OVERCONFIDENT": 0, "HALLUCINATION": 0})
        for r in settled:
            by_agent[r.agent_name][r.result_30d] += 1

        report = {}
        for agent, counts in by_agent.items():
            total = sum(counts.values())
            report[agent] = {
                "total_decisions": total,
                "pass_rate": counts["PASS"] / total,
                "overconfident_rate": counts["OVERCONFIDENT"] / total,
                "hallucination_rate": counts["HALLUCINATION"] / total,
            }

        # 排名: 幻觉率最高的 Agent → 需要降权或重训练
        ranked = sorted(report.items(), key=lambda x: x[1]["hallucination_rate"], reverse=True)

        return {
            "by_agent": report,
            "worst_agent": ranked[0][0] if ranked else "N/A",
            "worst_hallucination_rate": ranked[0][1]["hallucination_rate"] if ranked else 0,
            "total_decisions": len(settled),
            "global_pass_rate": sum(r.result_30d == "PASS" for r in settled) / len(settled),
        }
```

**验收标准**:
- [ ] 每条 AI 决策自动记录到 `HallucinationAuditor`
- [ ] 7 天和 30 天自动复盘
- [ ] 月度报告: 幻觉率 > 30% 的 Agent → 降权 50%
- [ ] 幻觉率 > 50% 的 Agent → 从决策链条中移除
- [ ] 季度报告: 幻觉率趋势图 + 各市场状态明细

---

## 三风险联合防御体系

### 全生命周期防御流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     实盘前 (Pre-Live)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 策略候选池                                                   │
│       ↓                                                         │
│  2. 搜索空间压缩 (相关性/经济逻辑/参数漂移)                        │
│       ↓                                                         │
│  3. CPCV + PBO + DSR 三重过拟合检测                               │
│     ├── VERDICT=APPROVED → 继续                                 │
│     ├── REJECTED → 归档                                         │
│     └── NEEDS_MORE_DATA → 观察队列                               │
│       ↓                                                         │
│  4. 衰减预算应用 → 预期实盘夏普 = 回测夏普 × β                    │
│       ↓                                                         │
│  5. 滑点建模 (Level-2 订单簿 → 真实成本)                         │
│       ↓                                                         │
│  6. WFO 增强版 (多制度覆盖 + Purge + Embargo)                     │
│       ↓                                                         │
│  7. Monte Carlo 鲁棒性检验 (Block Bootstrap)                     │
│       ↓                                                         │
│  8. 纸上交易 (30 个交易日)                                        │
│       ↓                                                         │
│  9. 绩效偏差 < 5% → 进入实盘                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     实盘中 (Live)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 决策权分离    │    │ 双 LLM 交叉   │    │ 事实锚定      │       │
│  │ LLM 只调 ±15% │    │ 验证         │    │ 每条陈述引用  │       │
│  │ Hub 做最终判断│    │ 矛盾 → HOLD  │    │ 具体数据源    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 拥挤度监控    │    │ 换手率控制    │    │ 自动熔断      │       │
│  │ 信号集中度 /  │    │ LoT 动态阈值  │    │ MaxDD > 15%  │       │
│  │ IC 衰减 /     │    │ 单次 ≤ 30%    │    │ → 策略暂停   │       │
│  │ 换手异常      │    │              │    │ MaxDD > 20%  │       │
│  │              │    │              │    │ → 全局减半   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     实盘后 (Post-Live)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 每日幻觉审计  │    │ 月度 Agent    │    │ 季度全面复盘  │       │
│  │ 决策 vs 7/30d │    │ 排名         │    │              │       │
│  │ 真实结果      │    │ 最低分 → 降权 │    │ β 系数更新   │       │
│  │              │    │ 或替换        │    │ 策略池清理    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │ 因子衰减检测  │    │ 绩效归因审计  │                           │
│  │ Page-Hinkley  │    │ Brinson +     │                           │
│  │ 渐变漂移检测  │    │ 因子归因      │                           │
│  └──────────────┘    └──────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 核心哲学

| 原则 | 说明 |
|------|------|
| **信任回测的方向，不信任量级** | 策略逻辑是否合理 → 可参考。具体数字 → 打折扣。 |
| **信任 LLM 的分析框架，不信任最终判断** | LLM 提供多维度视角 → 有价值。LLM 做买卖决策 → 不允许。 |
| **防御前置，而非事后补救** | CPCV/PBO 在实盘前执行。衰减预算在预期中体现。决策权分离在架构中固化。 |
| **可审计、可追溯、可降权** | 每条 AI 决策有审计记录。每个 Agent 有月度评分。低分 Agent 自动退出。 |

---

## 实施检查清单

### Phase 3-4 (回测引擎) 附加项

- [ ] `combinatorial_purged_cv()` 实现并集成到回测流水线
- [ ] `compute_pbo()` 实现并集成
- [ ] `compute_deflated_sharpe_ratio()` 实现并集成
- [ ] `validate_strategy()` 作为策略注册的强制入口
- [ ] `purged_walk_forward_validation()` 替代标准 WFO
- [ ] `AShareSlippageModel` 集成到 Backtrader 引擎
- [ ] 搜索空间压缩三层过滤实现

### Phase 5-6 (策略切换 + AI 引擎) 附加项

- [ ] `DecisionHub` 实现并作为唯一决策入口
- [ ] `FactAnchoredAgent` 实现 → 所有 Agent 继承此类
- [ ] `DualLLMValidator` 集成到 CIO 判决前
- [ ] `CrowdingMonitor` 集成到每日信号流水线
- [ ] `TurnoverController` (LoT 算法) 集成到仓位分配器

### Phase 7 (TUI 终端) 附加项

- [ ] 每个策略展示双数字: 回测夏普 / 预期实盘夏普
- [ ] 信号面板显示拥挤度指示器 (绿/黄/红)
- [ ] AI 推荐面板显示: 因子得分 / LLM 调整 / 置信度 / 安全检查结果
- [ ] 幻觉审计面板 (月度报告视图)

### Phase 8 (投前验证) 附加项

- [ ] `HallucinationAuditor` 集成 → 每日自动审计
- [ ] 衰减系数 β 自动更新脚本 (季度)
- [ ] 因子衰减检测定时任务 (Page-Hinkley)
- [ ] 月度 Agent 排名报告自动生成

---

**文档状态**: 完整 | **版本**: v1.0 | **关联**: `STRATEGY-AND-DEVELOPMENT.md` 的补充防御设计
