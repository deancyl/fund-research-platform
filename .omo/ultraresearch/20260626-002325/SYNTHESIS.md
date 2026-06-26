# Ultraresearch Synthesis: Production-Grade Backtesting Engine Architecture for Chinese Fund/Index Fund Strategies

Workers: 22+ queries · Waves: 2 · Sources: 50+ · Verifications: 0 (code verification deferred)

## Executive Summary

Building a production-grade backtesting engine for Chinese fund/index fund strategies requires a **hybrid architecture**: a vectorized engine for rapid research iteration (0.7s vs 14s+ for event-driven), feeding into an event-driven engine for execution-fidelity validation before deployment. The "two-framework stack" (VectorBT → pybroker/Backtrader) is the dominant industry pattern.

For Chinese A-share markets specifically, the engine must enforce **T+1 settlement**, **price limit up/down (±10%/±20%/±30%)**, **stamp duty (0.05% sell-side only)**, **minimum lot sizes (100/200 shares)**, and **transfer fees (~0.001%)**. These are non-negotiable constraints that most Western frameworks (Backtrader, Zipline) lack natively.

The most critical validation methodology is **walk-forward optimization with purged cross-validation** (López de Prado, 2018), combined with **Monte Carlo bootstrap confidence intervals** and the **Deflated Sharpe Ratio** to correct for multiple testing. The bootstrap.marketmaker.cc (2026) controlled study proves that bootstrap CIs for maximum drawdown are systematically optimistic — nominal 5% tails have 7-23% actual exceedance — meaning conservative safety margins are essential.

For Chinese-specific benchmarks, the CSI 300 (large-cap), CSI 500 (mid-small cap), and CSI All-Share index form the standard evaluation suite. The risk-free rate should use 10-year Chinese government bond yields or SHIBOR.

## Findings by Theme

### 1. Event-Driven vs Vectorized Architecture
- **Consensus**: Neither is superior — the correct architecture depends on the workflow phase.
- **Vectorized**: Processes entire arrays simultaneously via NumPy/numba. 10-100x faster than event-driven. VectorBT processes 500 tickers × 10 years in <1 second, identical results to event-driven engines. Ideal for research, parameter sweeping, and strategy screening.
- **Event-Driven**: Bar-by-bar sequential simulation enforcing real-world information constraints. Handles T+1, slippage, partial fills, limit orders, and intraday risk checks. Essential for pre-deployment validation. Backtrader takes 14.2s for the same 500-ticker × 5yr run.
- **Hybrid Workflow**: Vectorized research → Event-driven validation → Live deployment. This is the professional standard.
- **Key source**: InsiderFinance.io (2026), Interactive Brokers Campus, Systematic Long Short (2026).

### 2. A-Share/Chinese Market Rules
- **T+1 Settlement**: Shares bought today cannot be sold until T+1. This is mandatory and enforced at the exchange level.
- **Price Limits**: ±10% for Main Board, ±20% for ChiNext (300xxx) and STAR Market (688xxx), ±30% for Beijing Stock Exchange (8xxxxx/4xxxxx). Trades at limit prices may not execute (sealed limit-up/down).
- **Stamp Duty (印花税)**: 0.05% on sell-side only (reduced from 0.1% in August 2023). No stamp duty on buy-side.
- **Commission**: 万2.5-3 (0.025-0.03%) standard for retail, negotiable to 万1.5 for institutional. Minimum 5元 per trade.
- **Transfer Fee (过户费)**: ~0.001% both buy and sell sides, applies to SSE stocks.
- **Minimum Lot**: 100 shares (main board / ChiNext), 200 shares (STAR), with ±1 share increments on BSE/STAR.
- **Short Selling**: Highly restricted for retail; securities lending only.
- **SSE 2026 Revision**: New rules effective July 6, 2026. Extends post-market fixed-price trading to all A-shares/ETFs (was only STAR), adjusts risk-warning stock limits from 5% to 10%.
- **Implementation reference**: rock-mind/autoquant provides config-driven A-share rules; Leonard-Don/quant-trading-system commit #144 adds stamp_duty_rate, transfer_fee_rate, enforce_t_plus_1, price_limit_pct.

### 3. Framework Comparison
- **Backtrader** (14K ★): Event-driven, flexible OOP API, extensive docs. CRITICAL: Unmaintained since 2019. No Python 3.10+ guarantee. Single-threaded event loop is slow. No native A-share support. Best for learning, not new production work.
- **VectorBT** (4K ★): Vectorized, extreme speed (0.08s vs 2.3s Backtrader). Best for parameter sweeps/optimization. Limited order types, simple slippage, no live trading. Good for research, not deployment.
- **Zipline-Reloaded**: Quantopian heritage, Pipeline API for factor research. High setup cost (bundle system), no live trading. Best for equity factor research.
- **Qlib** (16K ★, Microsoft Asia): AI-native, 360+ built-in A-share factors (Alpha158/360), LightGBM/LSTM/Transformer models. Integrated data pipeline + backtest. No live trading. Best for ML factor mining.
- **VnPy** (24K ★): Full-stack platform, CTP interface for China futures. Most mature live trading ecosystem for Chinese markets. C++ core for performance. Best for China CTA + live trading.
- **PyBroker** (3.4K ★): Numba-accelerated vectorized engine. **Native walkforward analysis** with lookahead parameter. Bootstrap confidence intervals on all metrics (Sharpe, Profit Factor, MaxDD). BCa bootstrap for skewness correction. Supports AKShare (Chinese data). Best for ML strategy backtesting with robust validation.
- **Hikyuu**: C++ core with Python API. Has **TC_FixedA/TC_FixedA2015/TC_FixedA2017** transaction cost models specifically for Chinese A-share market history. Built-in A-share rules.
- **NautilusTrader**: Production-grade event-driven, HFT-capable, active development. Python/C++ hybrid. Most promising Backtrader alternative for 2026.
- **Strategy**: VectorBT for research → pybroker for walkforward/bootstrap validation → VnPy/NautilusTrader for live.

### 4. Walk-Forward Optimization
- **Rolling vs Anchored**: Rolling (sliding) windows forget old data, adapt to regime changes. Anchored (expanding) windows use all history, more data-efficient. Professional practice: anchored for ML training, rolling for backtest validation.
- **Purged K-Fold**: Removes training observations whose labels overlap test-set intervals (purge). Adds embargo buffer to prevent autocorrelation leakage. Implemented across 10+ OSS libraries (mlfinlab, skfolio, purged-cross-validation).
- **Combinatorial Purged CV (CPCV)**: Enumerates C(N,K) train/test splits. Produces a distribution of OOS Sharpe values, enabling Probability of Backtest Overfitting (PBO) estimation.
- **Deflated Sharpe Ratio (DSR)**: Corrects the observed Sharpe for the number of trials (parameter combinations tested). DSR > 0.95 = statistically significant after multiple-testing correction.
- **Walk-Forward Efficiency Ratio (WFER)**: Sum OOS PnL / Sum IS PnL. suenot/wfo-validity (2026) is the first controlled study testing WFER's folk thresholds (0.8/0.5/0.3).
- **PyBroker Implementation**: walkforward(windows=5, train_size=0.5, lookahead=1). The lookahead parameter prevents training data leakage across the test boundary.
- **Key Libraries**: eslazarev/purged-cross-validation (CPCV + DSR + PBO), neeljshah/walkforge (purged walk-forward + bootstrap CIs), suenot/wfo-validity (WFER controlled study), skfolio (CombinatorialPurgedCV + stationary bootstrap).

### 5. Performance Metrics Suite
| Metric | Formula | Good Threshold | Notes |
|--------|---------|---------------|-------|
| Sharpe Ratio | (Rp - Rf) / σp | >1.0 good, >2.0 excellent | Penalizes upside vol. Key limitations: normal distribution assumption, symmetric vol treatment |
| Sortino Ratio | (Rp - Rf) / σ_downside | >1.5 good | Fixes Sharpe's upside vol problem. Always >= Sharpe |
| Calmar Ratio | CAGR / |MaxDD| | >1.0 good, >2.0 excellent | Most intuitive: "return per dollar of worst drawdown" |
| Omega Ratio | ∫(F(x))dx above τ / below τ | >1.5-2.0 | Full distribution, threshold-aware. Best for asymmetric strategies |
| Ulcer Index | RMS of drawdown series | <5% comfortable | Measures drawdown depth AND duration |
| SQN | sqrt(N) × mean(expectancy) / std | >2.5 excellent | Van Tharp. Requires 30+ trades |
| Profit Factor | Gross Profit / Gross Loss | >1.5 confident | Simple, but ignores risk |
| Burke Ratio | Annualized Return / sqrt(ΣDD²) | Higher better | Like Calmar but considers ALL drawdowns, not just worst |
- **Implementation**: wraquant library offers 30+ metrics. PyBroker provides bootstrap CIs for each metric.

### 6. Monte Carlo Simulation
- **Key Insight (2026 Research)**: bootstrap.marketmaker.cc controlled study (6,000 experiments, 5 DGP families) proves: bootstrap CIs for maximum drawdown are **systematically optimistic**. Nominal 5% worst-case drawdown has 7-23% actual exceedance probability.
- **Method Comparison**: iid bootstrap (underestimates tail risk), block/stationary bootstrap (better, block size 5 for daily data), trade-level resampling.
- **BCa Bootstrap**: Bias-corrected and accelerated bootstrap corrects for skewness in return distributions. Used by PyBroker.
- **Minimum Iterations**: 1,000 for stable estimates, 5,000-10,000 for tail metrics (probability of ruin).
- **Key Outputs**: 95% CI for Sharpe ratio, 95% CI for maxDD, probability of negative Sharpe, probability of ruin (drawdown > X%).
- **Libraries**: suenot/bootstrap-coverage (controlled study code), arch (stationary_bootstrap), skfolio (stationary_bootstrap with automatic block length), PyBroker (BCa bootstrap built-in).

### 7. Slippage & Transaction Costs for Chinese Markets
- **Explicit Costs**:
  - Commission: 万1.5-3 (0.015-0.03%) per trade
  - Stamp Duty: 0.05% (sell-side only, 2023 rule)
  - Transfer Fee: ~0.001% both sides (SSE)
  - Min commission: 5元 per trade
- **Implicit Costs**:
  - Bid-Ask Spread: 1-3 bps (large cap), 3-10 bps (mid cap), 10-50 bps (small cap)
  - Market Impact: Linear model (MI = η × V/ADV) or square-root model (MI = α × σ × √(V/VT))
  - Typical η values: 0.05-0.10 (large cap), 0.10-0.20 (small cap)
- **Total One-Way Cost**: ~0.1-0.3% depending on trade size and liquidity
- **Total Round-Trip Cost**: ~0.2-0.6% — significant for high-turnover strategies
- **Hikyuu Cost Models**: TC_FixedA (commission 0.18% min¥5, stamp 0.1% sell, transfer 0.001 per share SSE), TC_FixedA2015 (updated 2015 rules), TC_FixedA2017 (2017 rules with Shenzhen transfer fee).
- **Vibe-Trading execution-model skill**: Detailed A-share cost tables with commission, stamp duty, spread estimates per market.

### 8. Bias Prevention
- **Survivorship Bias**: Include delisted companies + terminal returns (Shumway 1997). Academic estimates: 0.53%/month for Chinese private funds, 1-2%/year for equities. Use point-in-time constituent universes, not current index membership.
- **Look-Ahead Bias**: filing_date <= trade_date is the hard rule. Never use restated financials before their filing date. PIT architecture stores every data vintage — the backtest sees only what was available at decision time.
- **Solutions**: hyperDB (open pipeline, survivorship-free + PIT), ValueIn (111M+ facts, PIT + survivorship-free + append-only restatements), StockFit (PIT fundamentals API with restatement tracking).
- **Data Pipeline Structure**: Medallion architecture (Bronze=raw immutable → Silver=cleaned/PIT → Gold=analysis-ready). Parquet format with hive-style partitioning for efficient time-series queries.

### 9. Benchmark Selection for Chinese Funds
- **CSI 300** (000300.SH): Top 300 large-cap A-shares. All-sector benchmark. Tracks ~60% of A-share market cap. Corresponding ETF: 510300.SH.
- **CSI 500** (000905.SH): Stocks ranked 301-800 by market cap. Mid-small cap benchmark. Corresponding ETF: 512500.SH.
- **CSI All-Share / CSI A Share Index**: Full A-share universe benchmark.
- **Risk-Free Rate**: 10-year Chinese government bond yield, SHIBOR overnight/short-term rates.
- **ChinaBond Indices**: For fixed-income benchmarks (ChinaBond Comprehensive Index, ChinaBond Treasury Index).
- **Common Strategy Benchmark**: 95% CSI 300 + 5% bank deposit rate (for equity fund evaluation).
- **FOF Trends (2026)**: Chinese FOF benchmarks shifting from broad market indices to fund-of-fund indices (CSI Equity Fund Index, CSI Bond Fund Index) reflecting multi-asset allocation strategies.
- **Sector-Level Benchmarks**: CSI Financials, CSI Consumer Staples, CSI Technology for sector rotation strategies.

### 10. Data Pipeline Design
- **Architecture Pattern**: Bronze (raw, immutable) → Silver (cleaned, PIT-tagged, survivorship-free) → Gold (features, factors, analysis-ready).
- **PIT Enforcement**: Every data point carries effective_date (when the event occurred) and availability_date (when it became public knowledge). Backtests filter on availability_date.
- **Identity Management**: SCD2 (Slowly Changing Dimension Type 2) for security identity across ticker changes, mergers, name changes. Use permanent identifiers (PERMNO, composite FIGI).
- **Delisting Handling**: Track delisting events with reason codes. Impute terminal returns conservatively (bankruptcies → near zero, mergers → deal consideration). Apply Shumway (1997) delisting return adjustment.
- **Corporate Actions**: Pre-compute adjustment factors for splits, reverse splits, dividends, spin-offs. Maintain total return index (price + reinvested dividends).
- **Storage**: Parquet format, partitioned by dt=YYYY-MM-DD for efficient time-range queries.
- **Pipeline Tools**: hyperDB (open source, reproducible global equity pipeline), ValueIn (point-in-time fundamentals), athapar/financial-data-pipeline (SCD2, Polygon → BigQuery → dbt).
- **Incremental Updates**: Idempotent ingestion with watermark tracking for daily data updates (Polygon, AKShare for Chinese data).

## Verified Claims
No code verification was executed in this research. Key claims requiring empirical verification before production deployment:
1. Bootstrap CI coverage for MaxDD — verify against suenot/bootstrap-coverage DGP framework
2. PyBroker walkforward vs manual purged-K-fold implementation — correctness comparison
3. A-share transaction cost drag on high-turnover strategies (>10x monthly turnover)
4. CSI 300 vs CSI 500 benchmark sensitivity — regime-dependent outperformance

## Contradictions
1. **Backtrader maintenance**: Multiple 2026 sources disagree — some call it "aging/unmaintained" (python.financial, dev.to), while others note "moderate community activity" (misar.blog 2026). Objective: last commit was 2024, original author mementum last active 2022. Verdict: not recommended for new production work.
2. **VectorBT limitations**: Some sources say "limited to long-only, simple strategies" (python.financial), while others show complex multi-asset portfolio optimization (dev.to). The VectorBT PRO version adds significant capabilities. Free version is indeed limited.
3. **Survivorship bias magnitude**: Academic estimates range from 0.53%/month (Chinese private funds, Fama-French 5-factor) to 1-2%/year (US equities, Elton et al. 1996) to 2-4%/year (hedge funds). Estimate depends heavily on asset class and methodology.

## Gaps
1. **Chinese fund-specific backtesting**: Limited open-source implementations for index fund portfolio optimization. Most focus on individual stock strategies.
2. **Multi-period optimization**: Very few frameworks handle rebalancing optimization (frequency, threshold bands, tax-aware) specifically for Chinese fund context.
3. **Real-time integration**: Most frameworks are backtest-only. VnPy leads in live trading but has limited walk-forward/bootstrap validation.
4. **Transaction cost calibration**: A-share specific market impact parameters (η) need empirical calibration per market cap bucket and volatility regime.

## Expansion Trace
- Wave 1 (16 parallel queries): Covered all 10 axes. Sources: 50+ web pages, 20+ GitHub repos, 2 Context7 docs.
- Wave 2 (12 queries): Deep dives on CPCV (eslazarev, skfolio, mlfinlab), WFO validation (suenot/wfo-validity), bootstrap coverage (suenot/bootstrap-coverage), hikyuu cost models, pybroker/qlib docs, purged k-fold implementations.
- Leads expanded: 12/12. Leads closed: 10. Open leads: bias quantification for Chinese funds (limited academic data).
- Convergence: Zero new actionable leads after Wave 2. All 10 axes have at least one authoritative source and one reference implementation.
