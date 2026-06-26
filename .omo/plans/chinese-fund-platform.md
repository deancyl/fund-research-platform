# chinese-fund-platform - Work Plan

## TL;DR (For humans)

**What you'll get:**
一个专业的中国基金/指数基金 AI 投研平台。终端式界面（键盘驱动，类Wind/Choice风格），支持基金数据查询、20个策略（含历史胜率和夏普数据）、三层回测验证体系、市场状态自动识别与策略切换、以及基于多智能体辩论的 AI 投资建议。完全本地运行，无需付费数据源。

**Why this approach:**
经过5维度12方向60+来源的深度调研后确定：Python+Textual TUI（所有数据/量化库均为Python生态）、AKShare主力数据源（全免费+最全覆盖）、MyTT中国兼容技术指标（TA-Lib不兼容KDJ/MACD）、多智能体辩论+8因子评分的混合决策（量化+定性双重验证，LLM分析推理→确定性数学最终决策）。

**What it will NOT do:**
不是交易终端（无实盘下单功能）、不使用付费数据（100%免费数据源）、不支持海外市场（仅限中国基金/指数基金）、不是Web应用（纯终端TUI）、不包含个股推荐（专注基金和指数基金）。

**Effort:** XL (14周，8个Phase，60+功能模块)
**Risk:** Medium — 主风险为AKShare爬虫稳定性（已设计东方财富直接API备用+本地缓存容灾）
**Decisions I made for you:**
12个默认决策已记录在 `.omo/drafts/chinese-fund-platform.md`。核心包括：Python技术栈、AKShare数据源、SQLite存储、MyTT指标、CH-3因子模型、Textual TUI、红涨绿跌配色、8因子+LLM混合决策、Wind/Choice键盘驱动交互范式、DeepSeek+Ollama双LLM方案。

Your next move: 批准后使用 `/start-work` 开始 Phase 1 基础设施搭建。完整策略文档见 `.omo/docs/STRATEGY-AND-DEVELOPMENT.md`（含20个策略、胜率数据、三层验证架构、策略切换机制、风险管理体系）。

---

> TL;DR (machine): <1 line - effort, risk, deliverables>

## Scope
### Must have
### Must NOT have (guardrails, anti-slop, scope boundaries)

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: <TDD | tests-after | none> + framework
- Evidence: .omo/evidence/task-<N>-chinese-fund-platform.<ext>

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [ ] 1. <title>
  What to do / Must NOT do: <...>
  Parallelization: Wave <N> | Blocked by: <...> | Blocks: <...>
  References (executor has NO interview context - be exhaustive): <src/path:lines>
  Acceptance criteria (agent-executable): <exact command or assertion>
  QA scenarios (name the exact tool + invocation): happy + failure, Evidence .omo/evidence/task-1-chinese-fund-platform.<ext>
  Commit: <Y/N> | <type>(<scope>): <summary>

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
- [ ] F2. Code quality review
- [ ] F3. Real manual QA
- [ ] F4. Scope fidelity

## Commit strategy

## Success criteria
