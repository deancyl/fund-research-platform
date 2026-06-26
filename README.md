# 中国基金/指数基金 AI 投研平台

> **v0.1.0** — Headless Core 架构 | 311 单元测试 | 10 策略 | TUI + Web + CLI 三端

---

## 架构

```
Headless Core (零 UI 依赖)
├── data/         AKShare 数据采集 + SQLite/DuckDB 双引擎缓存
├── analysis/     8因子评分 · MyTT 技术指标 (KDJ/MACD/RSI/BOLL)
├── engine/       中国公募基金规则引擎
│   ├── 15:00 申赎截点  │  2026 阶梯赎回费  │  资金交收延迟
│   └── SmartRebalancer (FIFO调仓 + 赎回费惩罚)
├── strategy/     10 策略 (动量/均值回归/因子轮动/中国特色/组合)
│   └── 市场状态检测 + 策略切换引擎
├── recommend/    AI 决策引擎
│   ├── DecisionHub (6项安全检查)
│   ├── DebateEngine (多智能体辩论协议)
│   └── HallucinationAuditor (7/30天幻觉审计)
└── api.py        统一服务层入口
```

## 快速开始

```bash
# 安装依赖
pip install -e .

# 终端界面
python -m src.cli.main tui

# Web API
python -m src.cli.main web --port 8000

# 基金分析
python -m src.cli.main analyze 005827

# 运行测试
python -m pytest tests/ -v
```

## 策略库

| # | 策略 | 类型 | 夏普 | 胜率 |
|---|------|------|------|------|
| S1 | 因子动量轮动 | 动量 | 1.15 | 65% |
| S2 | ETF 多因子动量 | 动量 | 1.38 | 83.3% |
| S6 | PE/PB 估值分位带 | 均值回归 | — | 60-70% |
| S7 | RSI 均值回归+固收 | 均值回归 | — | 70%+ |
| S8 | 红利低波择时 | 均值回归 | 0.66 | — |
| S10 | 网格交易+Hurst | 均值回归 | — | 68% |
| S11 | 宏观四状态轮动 | 因子轮动 | 1.78 IR | 66% |
| S14 | 春季效应 | 因子轮动 | — | 80% |

## 开发进度

| Phase | 内容 | 进度 |
|-------|------|------|
| 1 | 基础设施 | ✅ 78% |
| 2 | 分析引擎 | ✅ 8因子评分 + 技术指标 |
| 3 | 回测引擎 | ✅ 中国规则引擎 |
| 4 | 策略库 | 🔄 10/20 (50%) |
| 5 | 市场状态 | ✅ 切换引擎 |
| 6 | AI 决策 | ✅ DecisionHub + DebateEngine |
| 7 | TUI 终端 | ✅ 基础实现 |
| 8 | Web 服务 | ✅ REST API |

**测试**: 311 passed · 0 failed

## 技术栈

Python 3.11 · Pydantic v2 · Polars · DuckDB · SQLite · Textual · FastAPI · MyTT · NumPy

## 文档

- [策略与开发文档](.omo/docs/STRATEGY-AND-DEVELOPMENT.md)
- [风险防御框架](.omo/docs/RISK-MITIGATION-FRAMEWORK.md)
- [系统开发契约](.omo/docs/SYSTEM-CONTRACT.md)
- [AI 执行指南](.omo/docs/EXECUTION-GUIDE.md)
