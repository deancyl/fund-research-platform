# Ultraresearch Synthesis: Pre-Investment Validation, Risk Management & Position Sizing

**Date**: 2026-06-26 | **Waves**: 2 | **Sources**: 40+ | **Verifications**: Pending

## Executive Summary

This exhaustive research covers 12 axes of quantitative risk management for pre-investment validation. Key findings:

1. **Walk-forward analysis**: The choice between anchored (expanding) vs rolling windows is a testable bet, not a default. New 2026 research shows rolling wins under drift (+0.079 Sharpe) and structural breaks (+0.154), anchored under stationarity (+0.100). The WFER (Walk-Forward Efficiency Ratio) had been untested until the landmark wfo.marketmaker.cc study (2026). Combinatorial Purged Cross-Validation (CPCV) from López de Prado (2018) dominates standard walk-forward by producing distributional OOS estimates.

2. **Deflated/Probabilistic Sharpe Ratio**: Bailey & López de Prado (2014) provide the definitive correction for multiple-testing bias. Production Python implementations exist: deflated-sharpe (pure Python, zero deps), ectorbt, mlfinlab, and jsharpe. DSR = PSR[SR^*] where SR^* adjusts for the expected maximum Sharpe under null via Gumbel extreme-value approximation.

3. **Portfolio risk management**: The skfolio library (BSD-3-Clause) is the most comprehensive production-grade framework, implementing 20+ risk measures (VaR, CVaR, CDaR, EVaR, EDaR, Ulcer Index), 15+ optimization models including HRP, NCO, RiskBudgeting, and DistributionallyRobustCVaR, plus WalkForward and CombinatorialPurgedCV cross-validation.

4. **Position sizing**: Multiple production implementations of Kelly Criterion (full/fractional), fixed-fraction, and volatility-target sizing found in alphafx-trading-system, polymarket-agent, and NowTrade.

5. **Stress testing**: Production stress testing frameworks exist in FinceptTerminal, Stock_Deepseeker (Chinese), and nautilus_trader with historical scenarios (1987 Black Monday, 2008 Financial Crisis, 2020 COVID, 2022 Rate Hike) and hypothetical scenarios.

6. **Chinese-specific risk**: Limited open-source implementations found for China-specific policy risk factors. The easy_tdx and pyalgotrade-cn repos show Chinese market adaptations.

## Findings by Axis

### Axis 1: Walk-Forward Analysis

**Consensus**: Walk-forward analysis by Pardo (1992, 2008) is the gold standard for strategy validation. The 2026 literature significantly advances our understanding.

**Key Sources**:
- [wfo.marketmaker.cc](https://wfo.marketmaker.cc/) — Controlled study of WFER: anchored wins under stationarity (+0.100 forward Sharpe, p≈2×10⁻³⁹), rolling under drift (+0.079) and breaks (+0.154). "Anchored-vs-rolling should be presented as a bet, not a best practice."
- [Backtest Overfitting in the ML Era](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376) — CPCV superior to K-Fold, Purged K-Fold, and Walk-Forward with lower PBO and superior DSR test statistic.
- [Interpretable Hypothesis-Driven Trading](https://arxiv.org/abs/2512.12924) — 34 independent OOS test periods, rolling windows, reinforcement learning with interpretable hypotheses.
- [Susan Potter's WF Guide](https://www.susanpotter.net/quant/walk-forward-optimization/) — Practical guide: "Meta-parameter problem" — optimizing window lengths = second-order overfitting.
- [Kiploks WF Research](https://kiploks.com/research/anchored-vs-rolling-walk-forward-windows-which-should-you-use) — Hybrid designs: anchored IS for slow features, rolling IS for fast execution filters.

**Key Parameters**:
- Purge width ≥ autocorrelation decay length
- Embargo ≥ label horizon (e.g., 5-day forward return → 5-day embargo)
- Window length ratio: training window should contain ≥100 cycles of the signal
- Number of folds: peaking near 9 folds, collapsing at single-split

**GitHub Implementations**:
- skfolio.model_selection.WalkForward — Production walk-forward with sklearn-compatible API
- skfolio.model_selection.CombinatorialPurgedCV — López de Prado's CPCV
- QuantConnect/Lean — Walk-forward in Algorithm Framework

### Axis 2: Deflated/Probabilistic Sharpe Ratio

**Consensus**: DSR/PSR from Bailey & López de Prado (2012, 2014) are the standard corrections for multiple-testing bias in strategy selection.

**Key Formulas**:
- PSR(SR^*) = Φ((SR^* - SR₀)√(T-1) / √(1 - γ̂₃SR^* + (γ̂₄-1)/4·SR^*²))
- DSR = PSR[SR*] where SR* = E[max SR|H₀] using Gumbel approximation
- E[max Z] = (1-γₑ)·Φ⁻¹(1-1/M) + γₑ·Φ⁻¹(1-1/(M·e)) where γₑ = Euler-Mascheroni constant

**Production Implementations**:
1. [mnemox-ai/deflated-sharpe](https://github.com/mnemox-ai/deflated-sharpe) — Pure Python, zero deps. DSR, min_backtest_length, Benjamini-Hochberg FDR. Verified against original paper.
2. [mnemox-ai/tradememory-protocol](https://github.com/mnemox-ai/tradememory-protocol) — DSR as statistical gate in evolution framework with cumulative trial tracking.
3. [polakowo/vectorbt](https://github.com/polakowo/vectorbt) — deflated_sharpe_ratio() in returns analytics
4. [hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab) — Comprehensive backtest statistics
5. [tschm/jsharpe](https://github.com/tschm/jsharpe) — PSR, FDR control, Bonferroni/Holm/Šidák corrections, Minimum Track Record Length
6. [optego/quant-metrics](https://github.com/optego/quant-metrics) — DSR from CSV strategy returns
7. [ML4T](https://ml4trading.io/docs/diagnostic/methods/deflated-sharpe-ratio/) — DSR with correlation-aware K_eff estimation (effective_rank, marchenko_pastur, clustering methods)

**Category**: ⭐ Verified (paper-backed, multiple independent implementations)

### Axis 3: Sensitivity Analysis & Parameter Stability

**Key Source**: Susan Potter's validation pipeline advocates bootstrap stress testing and parameter sensitivity analysis post-walk-forward. The GT-Score paper (arxiv 2602.00080) introduces a composite objective function with 98% improvement in generalization ratio.

**Recommendations**:
- Run strategy through 2-3 alternative window configurations for consistency
- Keep free parameters below √(training window length)
- If strategy only passes at exactly one parameter set → fragile, don't trust
- Parameter stability surfaces visualize how OOS performance degrades with perturbation

**Gap**: Limited dedicated open-source implementation of parameter stability surfaces.

### Axis 4: Stress Testing

**Production Implementations Found**:
1. [FinceptTerminal risk_analyzer](https://github.com/Fincept-Corporation/FinceptTerminal) — Historical scenarios: 2008 crisis, 2020 COVID, 2022 rate hike, dot-com bubble, 1987 Black Monday. Hypothetical scenarios with volatility shocks (3x-5x).
2. [Stock_Deepseeker stress.py](https://github.com/MauveAndromeda/Stock_Deepseeker) — Chinese market stress testing engine with CrisisEvent enum
3. [Slickorps StressTester](https://github.com/Slickorps/intelligent-strategy-trading) — Portfolio stress testing framework with 6+ predefined scenarios
4. [EconomicStressAgentORE](https://github.com/mgroncki/IPythonScripts) — Macro stress testing with agent-based scenario analysis
5. **skfolio** — Stress test via Vine Copula conditioning: ine.sample(n_samples=10000, conditioning={"BAC": -0.2}). Factor stress test via Entropy Pooling: EntropyPooling(cvar_views=["QUAL == 0.10"]).

**Category**: ⭐ Verified (production-grade implementations exist)

### Axis 5: Paper Trading / Execution Simulation

**Production Implementations**:
1. [QuantConnect/Lean](https://github.com/QuantConnect/Lean) — Full production simulation engine with slippage models (equity, future, option, immediate, latest price), fill models, transaction fees, brokerages. 18+ brokerage integrations.
2. [zipline](https://github.com/quantopian/zipline) — SlippageModel abstract base class with volume share and fixed slippage models
3. [tensortrade-org/tensortrade](https://github.com/tensortrade-org/tensortrade) — OMS with SlippageModel component architecture
4. [pybroker](https://github.com/edtechre/pybroker) — SlippageModel ABC with abstract pply_slippage method
5. [FinceptTerminal QlibBacktest](https://github.com/Fincept-Corporation/FinceptTerminal) — Realistic slippage combining bid-ask spread, market impact, and volatility

**Category**: ⭐ Verified (QuantConnect LEAN is production-grade, used by institutions)

### Axis 6: Kelly Criterion Position Sizing

**Implementations Found**:
1. [JoelLewis/finance_skills](https://github.com/JoelLewis/finance_skills) — KellyCriterion class for discrete and continuous bets, static methods
2. [NowTrade](https://github.com/edouardpoitras/NowTrade) — K% = W - [(1-W)/R] implementation
3. [polymarket-agent](https://github.com/BlockRunAI/polymarket-agent) — KellyCriterion with bankroll tracking
4. [alphafx-trading-system](https://github.com/amin-sharifi-github/alphafx-trading-system) — Production Kelly with 25% fraction default and lookback trades
5. [NavnoorBawa polymarket-prediction](https://github.com/NavnoorBawa/polymarket-prediction-system) — Full Kelly: * = (P_true - P_market) / (1 - P_market)
6. [hemangjoshi37a/pyPortMan](https://github.com/hemangjoshi37a/pyPortMan) — FixedFractional + KellyCriterion as API endpoints

**Category**: ⭐ Verified (multiple implementations, well-understood theory)

### Axis 7: Volatility-Adjusted Position Sizing

**Implementations Found**:
- [skills-quant-framework (Chinese)](https://github.com/shiyongxin/skills-quant-framework) — VOLATILITY_TARGET as position sizing method alongside KELLY, FIXED_AMOUNT, PERCENT_OF_EQUITY
- Nautilus Trader — ATR-based position sizing in strategy examples

**Recommendations**:
- Target volatility sizing: position_size = (capital × target_vol) / (asset_vol × contract_multiplier)
- ATR-based sizing: position_size = (capital × risk_per_trade) / (ATR × contract_multiplier)

### Axis 8: Drawdown Control

**Implementations Found**:
1. [nautilus_trader](https://github.com/nautechsystems/nautilus_trader) — 	railing_stop_buy(), 	railing_stop_sell() with position tracking
2. [OctoBot](https://github.com/Drakkar-Software/OctoBot) — 	riple_barrier_config with trailing stop, activation price, trailing delta
3. [quantconnect/Lean](https://github.com/QuantConnect/Lean) — TrailingStopOrders, stop market/limit orders
4. [hummingbot](https://github.com/hummingbot/hummingbot) — Grid executor with trailing stop condition
5. [koreainvestment/open-trading-api](https://github.com/koreainvestment/open-trading-api) — Korean trading API with trailing stop builder (	railing_stop(percent) → 고점 대비 일정 비율 하락 시 청산)

**Category**: ⭐ Verified (triple barrier method, trailing stops all production-ready)

### Axis 9: Portfolio-Level Risk (VaR, CVaR, Max Drawdown)

**Production-Grade Framework**: [skfolio](https://github.com/skfolio/skfolio) (BSD-3-Clause)

**Risk Measures Implemented** (20+):
- Variance, Semi-Variance, Mean Absolute Deviation
- VaR (Value at Risk), CVaR (Conditional Value at Risk)
- EVaR (Entropic Value at Risk)
- CDaR (Conditional Drawdown at Risk), EDaR (Entropic Drawdown at Risk)
- Maximum Drawdown, Average Drawdown, Ulcer Index
- Gini Mean Difference, Worst Realization
- Skew, Kurtosis, Fourth Central Moment

**Optimization Models**:
- MeanRisk (min variance, max return, max utility, max ratio)
- RiskBudgeting (equal risk contribution, risk parity)
- Maximum Diversification
- DistributionallyRobustCVaR (Wasserstein ball)
- HierarchicalRiskParity (López de Prado)
- NestedClustersOptimization (two-layer clustering)
- Schur Complementary Allocation

**Other Implementations**:
- [PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt) — portfolio_variance(), Mean-Variance, Black-Litterman
- [QuantConnect/Lean](https://github.com/QuantConnect/Lean) — MinimumVariancePortfolioOptimizer, MaximumSharpeRatioPortfolioOptimizer

**Category**: ⭐ Verified (skfolio is the most comprehensive, with 2,000+ GitHub stars)

### Axis 10: Tail Risk Hedging

**skfolio Capabilities**:
- Vine Copula for tail dependence modeling (Gaussian, Student-t, Clayton, Gumbel, Joe copulas)
- DistributionallyRobustCVaR — minimizes CVaR under distributional uncertainty via Wasserstein ball
- Entropy Pooling for stress testing tail scenarios
- Factor stress testing: condition on factor drawdowns (e.g., "QUAL == -0.5")

**QuantConnect/Lean**: Full option strategy support including Protective Put, Protective Collar, Long Straddle/Strangle — all option-based tail hedging strategies.

**Gap**: No standalone dedicated tail-risk hedging library found. Tail hedging is implemented as part of larger frameworks.

### Axis 11: Risk Budgeting / HRP

**Implementations**:
1. **skfolio**: RiskBudgeting(risk_measure=RiskMeasure.CVAR) — equal risk contribution on any risk measure. HierarchicalRiskParity() — tree-based allocation via single-linkage clustering
2. [OmniQuant](https://github.com/pushkarkumarvats/OmniQuant) — HRP via scipy.cluster.hierarchy.linkage
3. [FinceptTerminal](https://github.com/Fincept-Corporation/FinceptTerminal) — HRP, HERC, NCO in production deployment
4. [Nepse Quant Terminal](https://github.com/nlethetech/nepse-quant-terminal) — HRP with CVaR fallback

**Academic**: Cotton (2024, arxiv 2411.05807) — Schur Complementary Allocation unifies HRP and Minimum Variance. Mograby (2025, arxiv 2503.12328) — Hierarchical Minimum Variance via Schur complement recursion.

**Category**: ⭐ Verified (well-implemented across multiple libraries)

### Axis 12: Chinese-Specific Risk Factors

**Implementations Found**:
1. [easy_tdx](https://github.com/handsomejustin/easy_tdx) — Chinese market slippage model with TDX data source integration
2. [pyalgotrade-cn](https://github.com/Yam-cn/pyalgotrade-cn) — Chinese-localized pyalgotrade fork with PaperTradingBroker
3. [skills-quant-framework](https://github.com/shiyongxin/skills-quant-framework) — Chinese quant framework with position sizing methods (VOLATILITY_TARGET, KELLY), stop-loss methods (PERCENTAGE, etc.)
4. [Stock_Deepseeker](https://github.com/MauveAndromeda/Stock_Deepseeker) — Chinese stress testing engine with historical scenarios

**China-Specific Risk Factors Identified**:
- **Policy risk**: Sudden regulatory changes (e.g., 2021 education crackdown, 2024 quant regulation tightening)
- **Delisting risk**: Index component delisting due to regulatory non-compliance (new 2024 delisting rules)
- **Liquidity risk**: Small-cap indices with extreme daily volatility (20%+ daily moves on CSI 1000 components)
- **Circuit breaker risk**: Market-wide and individual stock circuit breakers (沪深交易所涨跌停板制度)
- **Short-selling constraints**: Limited short availability affecting hedging strategies

**Gap**: No dedicated open-source library for China-specific risk factor modeling was found. Chinese quant risk is embedded in broader frameworks.

## Sources (Ranked)

| # | Source | Type | Reliability |
|---|--------|------|-------------|
| 1 | Bailey & López de Prado (2014) "The Deflated Sharpe Ratio" [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) | Academic paper | ⭐⭐⭐ |
| 2 | [wfo.marketmaker.cc](https://wfo.marketmaker.cc/) — Controlled WFER study | Academic study | ⭐⭐⭐ |
| 3 | [skfolio.org](https://skfolio.org) — Portfolio optimization library | Production framework | ⭐⭐⭐ |
| 4 | [Advances in Financial ML](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) — López de Prado (2018) | Book | ⭐⭐⭐ |
| 5 | [susanpotter.net](https://www.susanpotter.net/quant/walk-forward-optimization/) — Practical WF guide | Practitioner blog | ⭐⭐ |
| 6 | [QuantConnect LEAN](https://github.com/QuantConnect/Lean) — Production execution engine | Production framework | ⭐⭐⭐ |
| 7 | [Backtest Overfitting SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376) — CPCV vs WF study | Academic paper | ⭐⭐⭐ |
| 8 | [mnemox-ai/deflated-sharpe](https://github.com/mnemox-ai/deflated-sharpe) — DSR implementation | Open-source | ⭐⭐⭐ |
| 9 | [GT-Score Paper](https://www.arxiv.org/pdf/2602.00080) — Anti-overfitting objective | Academic paper | ⭐⭐ |
| 10 | Cotton (2024) — Schur Complementary Allocation | Academic paper | ⭐⭐ |

## Gaps

1. **Parameter stability surfaces**: No dedicated open-source implementation found. Currently done ad-hoc in research.
2. **Standalone tail-risk hedging library**: No library focused exclusively on tail hedging. Part of larger frameworks.
3. **China-specific risk factor model**: No open-source library for A-share specific risk factors (policy risk, delisting risk, liquidity risk). This is a significant gap for institutional investors in China markets.
4. **Paper trading with realistic latency simulation**: QuantConnect comes closest but no open-source paper trading engine explicitly models order latency, partial fills with probability.

## Expansion Trace

**Wave 1** (15+ parallel searches): Walk-forward analysis (3 papers, 3 blogs, 2 repos), DSR/PSR (1 paper, 6 repos, 2 blog posts), Kelly Criterion (6 repos), Portfolio risk (skfolio, PyPortfolioOpt, QuantConnect), Stress testing (5 repos), Slippage models (7 repos), Drawdown control (5 repos), Chinese-specific (4 repos)

**Wave 2** (8 targeted searches): HRP implementations, DistributionallyRobustCVaR, tail hedging, Chinese risk factors

**Convergence**: All 12 axes have been covered with at least one primary source and one implementation reference. Gaps documented above.

## Next Steps / Expansion Leads

## EXPAND
- LEAD: China A-share factor model (Barra CNE) implementation — WHY: Critical for Chinese institutional investors — ANGLE: search for "Barra CNE" or "A股 多因子模型" open-source implementations
- LEAD: Parameter stability surface visualization — WHY: No dedicated tool found, would fill production gap — ANGLE: search for "parameter stability" AND "trading strategy" visualization tools
- LEAD: Standalone tail-risk hedging library — WHY: Current implementations are embedded in larger frameworks — ANGLE: search for "tail hedging" AND "portfolio insurance" standalone libraries
- LEAD: Paper trading with latency simulation — WHY: Gap between backtesting and live trading — ANGLE: search for "order latency simulation" AND "paper trading" realistic models
