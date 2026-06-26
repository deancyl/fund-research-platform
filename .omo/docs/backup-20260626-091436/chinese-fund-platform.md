---
slug: chinese-fund-platform
status: awaiting-approval
intent: unclear
pending-action: write .omo/plans/chinese-fund-platform.md
approach: |
  Python + Textual TUI professional terminal platform for Chinese fund/index fund investment research.
  Keyboard-driven (Wind/Choice paradigm), multi-panel layout, with proactive investment recommendations.
  Architecture: Data Layer (AKShare+SQLite) → Analysis Engine (MyTT+fincore+jh-factors) → 
  Recommendation Engine (8-factor scoring + multi-agent LLM debate) → Textual TUI.
  Entirely local, offline-first, no paid API dependencies.
---

# Draft: chinese-fund-platform

## Components (topology ledger)
| id | outcome | status | evidence path |
|---|---|---|---|
| C1 | Data Acquisition Layer — fetches fund NAV, index data, news from AKShare/Eastmoney, caches to SQLite | active | `src/data/` |
| C2 | Quantitative Analysis Engine — technical indicators, risk metrics, factor models, backtesting | active | `src/analysis/` |
| C3 | News & Sentiment Engine — news aggregation, Chinese NLP sentiment (bardsai + SnowNLP + jieba) | active | `src/news/` |
| C4 | Fund Scoring & Recommendation Engine — 8-factor scoring, multi-agent LLM debate, ranking | active | `src/recommend/` |
| C5 | Terminal UI — Textual-based professional TUI with keyboard commands, multi-panel, charts | active | `src/tui/` |
| C6 | MCP Server — exposes platform capabilities as MCP tools for AI agent integration | deferred | `src/mcp/` |

## Open assumptions (announced defaults)
| # | assumption | adopted default | rationale | reversible? |
|---|---|---|---|---|
| A1 | Tech stack language | Python 3.11+ | All core data libs (AKShare, MyTT, fincore, xalpha) are Python-only. Textual is the most mature Python TUI with fintech examples. | Yes (Go/Rust alternatives exist but lose Python lib ecosystem) |
| A2 | Primary data source | AKShare + direct Eastmoney APIs | Free, no token, most comprehensive Chinese fund data (19K+ stars, 185 releases). Baostock as index fallback. | Yes (Tushare or other sources can be added) |
| A3 | Local storage | SQLite | Zero setup, portable, sufficient for personal research platform. No PostgreSQL required. | Yes (can upgrade to DuckDB/PostgreSQL) |
| A4 | Technical indicator library | MyTT (TDX-compatible) + pandas-ta (generic) | MyTT produces identical results to 通达信/同花顺 (KDJ, MACD). TA-Lib is incompatible with Chinese brokers. | Yes (ta_cn alternative) |
| A5 | Chinese factor model | CH-3 (jh-factors) | CH-3 model removes bottom 30% market cap to eliminate shell-value contamination, validated for A-shares. | Yes (FF5, SY4 alternatives available) |
| A6 | Sentiment model | bardsai/finance-sentiment-zh-fast (96.1% accuracy) + SnowNLP fallback | Pre-trained financial Chinese sentiment model, 264 samples/s on GPU. SnowNLP as offline CPU fallback. | Yes (other HuggingFace models) |
| A7 | Recommendation method | 8-factor scoring + multi-agent LLM debate | Combines quantitative scoring (fundseeker-style) with qualitative debate (openInvest-style). | Yes (weights and agents configurable) |
| A8 | TUI layout paradigm | Wind/Choice-style: keyboard command palette + F5/F9 shortcut system + multi-panel | Proven professional terminal UX. Users expect this interaction model. | Yes (layouts are configurable) |
| A9 | Chart rendering | plotext (via Textual integration) | plotext supports candlestick charts, line charts, bar charts natively in terminal. | Yes (matplotlib with terminal rendering) |
| A10 | Color convention | Red = up (涨), Green = down (跌) | Chinese market convention (opposite of Western). | Yes (configurable) |
| A11 | LLM for recommendations | DeepSeek API (local Ollama fallback) | DeepSeek is cheapest Chinese-capable LLM. Ollama for fully offline mode. | Yes (any OpenAI-compatible API) |
| A12 | Offline-first | Full local operation with cached data; network only for data refresh | User requested "本地" (local). Must work without internet after initial data sync. | No (core requirement) |

## Findings (cited - path:lines)
| # | finding | source |
|---|---|---|
| F1 | AKShare (akfamily/akshare, 19.8K★) is the most comprehensive free Chinese financial data library. Covers fund NAV (`fund_open_fund_daily_em`), ETF realtime (`fund_etf_spot_em`), index composition (`index_stock_cons_csindex`), fund holdings (`fund_portfolio_hold_em`). | librarian: bg_5c4adca6 |
| F2 | TA-Lib is INCOMPATIBLE with Chinese broker indicators: STOCH() lacks J-line, RSI can exceed 100, MACD uses different SMA algorithm. MyTT provides TDX-compatible KDJ, MACD, BOLL in ~100 lines pure Python. | librarian: bg_10b5fbae |
| F3 | Existing open-source projects: iFund (fund clustering + MCP), YMOS (SOP + soft routing), openInvest (multi-agent LLM committee), TradingAgents-astock (7-analyst debate). All are Python-based with AKShare data. | librarian: bg_f0e6d128 |
| F4 | Wind/Choice professional terminals are defined by: keyboard command palette (keyboard elf), F5 (chart) + F9 (deep data) paradigm, multi-panel workspace, data browser (no-code SQL), portfolio P&L monitoring. No existing open-source terminal replicates this for Chinese funds. | librarian: bg_67455506 |
| F5 | Textual (26K★) is the best Python TUI for this use case: FinTerm and StocksTUI prove the concept, DataTable widget supports keyboard navigation, Worker system handles async data fetching, CSS theming enables dark theme + red/green coloring. CJK wcwidth integration merged Feb 2026. | librarian: bg_0191d961 |
| F6 | jh-factors provides CH-3 (Chinese 3-factor) model specifically designed for A-shares, removing shell-value contamination by excluding bottom 30% market cap stocks. | librarian: bg_10b5fbae |
| F7 | fundseeker (yuna78/fundseeker) implements 8-factor scoring with configurable weights for Chinese mutual funds. Morningstar MRAR methodology uses loss-aversion penalty and 5-star rating distribution (10%/22.5%/35%/22.5%/10%). | librarian: bg_10b5fbae |
| F8 | bardsai/finance-sentiment-zh-fast achieves 96.1% accuracy on Chinese financial sentiment (3-class: positive/negative/neutral). SnowNLP requires retraining with financial corpus for acceptable results. | librarian: bg_10b5fbae |
| F9 | Longbridge Terminal (907★, Rust+Ratatui) is the closest existing open-source TUI trading terminal but targets HK/US/CN stocks, not Chinese funds. Proves TUI terminal UX is viable. | librarian: bg_f0e6d128 |
| F10 | All existing Chinese fund platforms are "passive query" (search → display). None provide proactive investment recommendations. This is the key differentiator opportunity. | librarian: bg_f0e6d128 |
| F11 | Direct Eastmoney APIs provide real-time fund valuation (`fundgz.1234567.com.cn/js/{code}.js`), fund lists, and historical NAV without any registration. Can serve as fallback when AKShare breaks. | librarian: bg_5c4adca6 |
| F12 | VectorBT is 20x-100x faster than Backtrader for parameter optimization but lacks A-share T+1/limit-up-down rules. Backtrader is recommended for final validation with Chinese market rules. | librarian: bg_10b5fbae |

## Decisions (with rationale)
| # | decision | rationale |
|---|---|---|
| D1 | Monorepo structure: `src/data/`, `src/analysis/`, `src/news/`, `src/recommend/`, `src/tui/`, `src/cli/` | Clean separation by concern, each independently testable |
| D2 | SQLite schema: `funds`, `fund_nav`, `fund_holdings`, `indexes`, `index_constituents`, `news_articles`, `recommendations`, `portfolio` | Normalized schema covering all data domains |
| D3 | Configuration via YAML (`config.yaml`): watchlist symbols, factor weights, LLM settings, update intervals | Simple, human-readable, no code changes for customization |
| D4 | CLI entry point with subcommands: `fund-research tui` (launch TUI), `fund-research update` (data sync), `fund-research analyze --code 005827` (single fund analysis), `fund-research recommend` (generate recommendations), `fund-research serve` (MCP server) | Multiple interaction modes: interactive TUI, batch analysis, AI agent |
| D5 | Cache-first architecture: always check SQLite before network request. Stale-while-revalidate for near-realtime data | Offline capability + freshness |
| D6 | Recommendation engine runs on schedule (daily after market close) and on-demand | Balances freshness with API rate limits |
| D7 | MCP server is deferred (C6) but architecture planned for: all core analysis functions exposed as tools | Future-proofing for AI agent integration without architecture change |

## Scope IN
- Fund NAV querying (domestic mutual funds, ETF, LOF, money market)
- Index data: CSI 300, CSI 500, CSI 1000, ChiNext, STAR 50, sector indices
- Technical indicators: KDJ, MACD, BOLL, RSI, MA, volume (China-compatible algorithms)
- Fund multi-factor scoring (8-factor: return, trend, risk-adjusted, drawdown, manager, rating, size, momentum)
- Investment recommendation engine (quantitative scoring + LLM qualitative analysis)
- News aggregation from Chinese financial sources
- News sentiment analysis (Chinese NLP)
- Keyboard-driven TUI with command palette (Wind/Choice style)
- Multi-panel layout: watchlist, chart, news feed, recommendation panel
- F5 (market data) + F9 (deep analysis) shortcut system
- Fund comparison (side-by-side: performance, holdings, risk metrics)
- Portfolio tracking with P&L calculation
- Index fund backtesting (DCA, momentum rotation, PE-band strategy)
- Data export: CSV, Excel
- Dark theme with Chinese market colors (red up, green down)
- Offline-first: full functionality without network after data sync
- Interactive keyboard navigation (Vim-style: j/k/h/l, / for search)
- Custom watchlist management
- Configuration via YAML file

## Scope OUT (Must NOT have)
- Live trading / order execution (research platform only, NOT a trading terminal)
- Real-time Level-2 market data (requires paid exchange subscription)
- Mobile app or web interface (terminal TUI only)
- User authentication / multi-user support (single-user local tool)
- Cloud deployment / SaaS (local-first by design)
- Foreign market data (US, HK, JP, etc. — China only)
- Individual stock picking/recommendation (funds and index funds only)
- High-frequency trading strategies
- Social features / sharing
- Paid API dependencies (100% free data sources)
- Copying existing open-source code verbatim (reference only)
- Automated trading bot / alert system

## Open questions
(none — all resolved via research)

## Approval gate
status: awaiting-approval

### Brief for User

**What I found:**
- AKShare is the undisputed best free data source for Chinese funds (19.8K★, covers fund NAV, ETF, index composition, fund holdings)
- MyTT is essential for China-compatible technical indicators (TA-Lib produces wrong KDJ/MACD for Chinese markets)
- No existing open-source project replicates the Wind/Choice terminal experience for Chinese funds — this is a genuine gap
- Textual (Python TUI) has proven financial terminal examples (FinTerm, StocksTUI) and recently merged CJK width fixes
- The key differentiator — proactive recommendations — is absent from all existing platforms

**Approach I intend to plan:**
Python + Textual TUI, 6-component architecture, keyboard-driven professional terminal with 8-factor scoring + LLM debate engine generating investment recommendations. Entirely local, free data sources, offline-first.

**12 adopted defaults** (recorded in draft) — veto any at the gate. Key ones: Python stack, AKShare data, MyTT indicators, CH-3 factor model, Textual TUI, DeepSeek LLM with Ollama fallback, Wind/Choice command palette UX.

**Ready to write the complete work plan.** Approve to proceed?
