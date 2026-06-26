# Ultraresearch Wave-1 Journal
Generated: 2026-06-26 00:25

## Axis 1: Event-driven vs Vectorized Backtesting
- Sources: insiderfinance.io, ibkr.com, dev.to, asmr.education, systematiclongshort.com
- Key finding: Hybrid workflow (vectorized screening -> event-driven validation) is industry standard
- Vectorized: 0.7s vs Event-driven: 14.2s (20x gap), identical correctness
- VectorBT can process 500 tickers x 10yr in <1 sec
- Professional firms use BOTH: vectorized for research, event-driven for pre-deployment

## Axis 2: A-share/Chinese Market Rules
- T+1 settlement (mandatory)
- Price limits: +/-10% (main), +/-20% (ChiNext/STAR), +/-30% (BSE from July 2026)
- Stamp duty: 0.05% sell-side only (2023 rule)
- Commission: 0.025% (wan2.5) both ways
- Transfer fee: 0.001%
- Min lot: 100 shares (main board), 200 shares (STAR), +/-1 share increments on BSE/STAR
- SSE Trading Rules 2026 Revision effective July 6, 2026

## Axis 3: Frameworks Comparison
- Backtrader: 14K stars, event-driven, unmaintained since 2019
- VectorBT: 4K stars, vectorized, 10-100x faster
- Zipline: community fork, Pipeline API, no live trading
- Qlib: 16K stars, Microsoft Asia, 360+ A-share factors
- VnPy: 24K stars, full-stack, CTP interface
- pybroker: 3.4K stars, Numba accelerated, built-in walkforward+bootstrap
- Hikyuu: C++ core, Chinese market cost models (TC_FixedA)

## Axis 4: Walk-forward Optimization
- Rolling vs anchored (expanding) windows
- Purged K-fold: purge + embargo to prevent temporal leakage
- CPCV: C(N,K) combinatorial train/test splits for PBO estimation
- Deflated Sharpe Ratio for multiple-testing correction
- WFER: OOS PnL / IS PnL efficiency ratio
- pybroker has native walkforward() with lookahead param
- Libraries: purged-cross-validation, walkforge, wfo-validity

## Axis 5: Performance Metrics
- Sharpe: >1.0 good, >2.0 excellent (penalizes upside vol)
- Sortino: downside-only, always >= Sharpe
- Calmar: CAGR/|MaxDD|, >1.0 good
- Omega: full distribution, threshold-aware
- Ulcer Index: RMS of drawdowns, <5% comfortable
- SQN: Van Tharp scale <1.6 insufficient, >2.5 excellent
- wraquant has 30+ metrics

## Axis 6: Monte Carlo Simulation
- Block bootstrap preserves autocorrelation (block size 5 for daily)
- BCa bootstrap corrects for skewness
- 1,000 iterations min, 5,000-10,000 for tail metrics
- Key research: bootstrap CIs for maxDD are systematically optimistic
- suenot/bootstrap-coverage: controlled CI coverage study

## Axis 7: Slippage and Costs
- A-share commission: wan1.5-3 (2-3bps)
- Stamp duty: 0.05% sell-side only
- Transfer fee: ~0.001% both sides
- Bid-ask spread: 1-3bps large cap, 3-10bps mid cap
- Market impact: linear (MI=eta x V/ADV) and square-root models
- Hikyuu TC_FixedA: commission 0.18% min Y5, stamp tax 0.1% sell

## Axis 8: Bias Prevention
- Survivorship bias: include delisted, use point-in-time datasets
- Look-ahead: filing_date <= trade_date
- PIT architecture: store every data vintage
- Academic: survivorship bias 0.53%/month private funds, 1-2%/year equities
- Look-ahead can inflate returns 100-500bps

## Axis 9: Benchmark Selection
- CSI 300: top 300 large-caps (core A-share benchmark)
- CSI 500: mid-small cap 500 stocks
- CSI All-Share: full A-share universe
- Risk-free: 10yr Chinese govt bond yield, SHIBOR
- Common: 95% CSI 300 + 5% bank deposit rate
- ChinaBond indices for fixed income

## Axis 10: Data Pipeline
- Medallion architecture (bronze/silver/gold)
- PIT enforcement: filing_date filter, SCD2 for identity
- Delisted securities: Shumway (1997) delisting returns
- Parquet format, hive-style partitioning
- hyperDB: open pipeline for global equity data
- ValueIn: 111M+ facts, PIT + survivorship-free
