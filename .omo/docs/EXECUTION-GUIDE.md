# AI 代理执行指南 — 原子化切片喂养策略

> **适用对象**: 本地 AI 代码代理（DeepSeek-R1 / DeepSeek-V4 / Qwen2.5-Coder）
> **核心原则**: 永远不要让 AI 一次性看到全部架构文档。信息密度过载 → 骨架偷懒（`# TODO`）+ 契约降级（顺手在视图层写计算逻辑）。
> **每次只喂一个 Phase 中的一个原子任务。禁止跨层、禁止跳跃。**

---

## 模型选择建议

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **架构骨架 + 数据 Schema** | Qwen2.5-Coder-32B | 对 Pydantic/SQL 的结构化代码生成精准，严格执行类型约束 |
| **中国规则引擎（15:00截点/赎回费/CashLock）** | DeepSeek-V4 或 DeepSeek-R1 | 复杂状态机逻辑 + 金融规则硬编码，需要强推理链 |
| **策略实现 + 回测引擎** | DeepSeek-V4 | 数值计算密集，NumPy/Numba 代码质量高 |
| **TUI 适配器 (Textual)** | Qwen2.5-Coder-32B | CSS/布局代码更适合前端型模型 |
| **Web 适配器 (FastAPI + Vue)** | DeepSeek-V4 | 全栈能力均衡 |
| **LLM Agent Prompts** | DeepSeek-R1 (满血) | 需要极强的语言理解和结构输出能力 |

> **本地部署**: Ollama 拉取 `deepseek-r1:70b` 或 `qwen2.5-coder:32b`。最小可用: `deepseek-coder-v2:16b`（速度优先，牺牲少量推理质量）。

---

## 原子化切片序列（严格按此顺序喂养）

每次对话只给 AI **一个文件** 的上下文 + **一个明确任务**。完成后验证 → 再进入下一个切片。

---

### 切片 0: 立规矩（10 分钟）

**喂入**: `SYSTEM-CONTRACT.md` （全文）
**指令**:
> "请阅读系统契约。不要写任何代码。用一段话复述你的理解：Headless Core 的边界在哪？场外公募赎回费的三档阶梯是什么？"

**验证标准**: AI 的复述必须精确覆盖 9 个约束点。如果遗漏 → 重新阅读直到完整。

---

### 切片 1: 项目骨架（20 分钟）

**喂入**: `STRATEGY-AND-DEVELOPMENT.md` §1.2（目录结构）
**指令**:
> "根据 Headless Core 架构，创建完整的分层目录结构。所有文件只需包含合法的 `__init__.py` 和模块级 docstring。禁止在 `src/core/` 的任何文件中导入 `textual`、`fastapi`、`flask`。禁止写 TODO 或 pass 占位符。"

**验证标准**:
```bash
# 检查零 UI 泄漏
grep -r "from textual" src/core/ && echo "❌ FAIL" || echo "✅ PASS"
grep -r "from fastapi" src/core/ && echo "❌ FAIL" || echo "✅ PASS"
grep -r "# TODO" src/core/ && echo "❌ FAIL" || echo "✅ PASS"
```

---

### 切片 2: 数据 Schema（30 分钟）

**喂入**: `STRATEGY-AND-DEVELOPMENT.md` §4.3.1（FundChannel + FundTradingProfile）+ 审计中 PositionLot schema
**指令**:
> "实现 `src/core/data/schema.py`。使用 Pydantic v2 BaseModel。必须包含:
> - FundChannel 枚举（4 个通道）
> - FundTradingProfile（含 cutoff_time, redemption_fee_schedule, settlement_delay_days, max_subscription_per_day_yuan, is_suspended）
> - PositionLot（purchase_date, shares, purchase_nav, cost_amount — 这是 FIFO 追踪的核心）
> - FundPosition（fund_code, channel, lots: List[PositionLot], current_nav, market_value, weight_pct）
> 所有字段必须使用 `Field(description=...)` 注释。"

**验证标准**:
```python
from src.core.data.schema import PositionLot, FundPosition
from datetime import date
# 测试 FIFO: 两次买入同一基金
lot1 = PositionLot(purchase_date=date(2026,3,1), shares=1000, purchase_nav=1.5, cost_amount=1500)
lot2 = PositionLot(purchase_date=date(2026,5,1), shares=500, purchase_nav=1.8, cost_amount=900)
pos = FundPosition(fund_code="005827", channel="OTC_OPEN_END", lots=[lot1, lot2], current_nav=2.0, total_shares=1500, market_value=3000, weight_pct=0.15)
# 验证: lots 按日期排序 → FIFO 正确
assert pos.lots[0].purchase_date < pos.lots[1].purchase_date
```

---

### 切片 3: 中国规则引擎 — 三个生死线（60-90 分钟）

**⚠️ 这是整个系统最核心的切片。只允许一个模块一个模块地喂。**

#### 3a: 15:00 申赎截点

**喂入**: `STRATEGY-AND-DEVELOPMENT.md` §4.3.2（OrderCutoffValidator 完整代码）
**指令**:
> "实现 `src/core/engine/order_cutoff.py`。完全按照给出的代码规范编写。特别关注:
> 1. `resolve_execution_date()` — 15:00 前 vs 后的逻辑必须精确
> 2. `validate_backtest_order()` — 前瞻偏差检测（盘后信号 + 当日净值 = 拒绝）
> 3. 包含完整 type hints、docstring、和 3 个 pytest 单元测试"

#### 3b: 2026 阶梯赎回费 + FIFO

**喂入**: `STRATEGY-AND-DEVELOPMENT.md` §4.3.3（RedemptionFeeCalculator） + 审计中的 FIFO 逻辑
**指令**:
> "实现 `src/core/engine/redemption_fee.py`。关键要求:
> 1. 阶梯费率表必须精确: <7天 1.5%, 7-30天 1.0%, 30-180天 0.5%, ≥180天 0%
> 2. 区分场内 ETF (channel=ETF_ON_EXCHANGE → 0 费) 和场外公募
> 3. 包含 FIFO 持有期计算: 接收 `List[PositionLot]` + `current_date` → 计算每批的持有天数
> 4. `enforce_min_hold()` — 持有 < 7 天返回 RED 警告，< 30 天返回 YELLOW
> 5. 包含至少 5 个 pytest 测试: 持有 3 天(1.5%)/持有 15 天(1.0%)/持有 60 天(0.5%)/持有 200 天(0%)/场内ETF(始终0%)"

#### 3c: 资金交收延迟状态机

**喂入**: `STRATEGY-AND-DEVELOPMENT.md` §4.3.4（CashLockManager 完整代码）
**指令**:
> "实现 `src/core/engine/cash_lock.py`。必须:
> 1. 使用 `collections.deque` 作为 LockedCash 队列
> 2. 交收延迟: ETF=0天, OTC=4天, ETF_FEEDER=3天, QDII=8天
> 3. `lock()` — 赎回时立即冻结资金
> 4. `unlock_daily()` — 每个交易日开始时释放到期资金
> 5. `get_buying_power()` — 返回当前可用购买力
> 6. 禁止在 lock/unlock 中使用任何数据库查询 → 纯内存操作
> 7. 包含测试: 模拟连续 3 次场外赎回 → 验证第 4 天资金逐步解冻"

---

### 切片 4: 智能调仓 — SmartRebalancer（45 分钟）

**喂入**: 审计中 SmartRebalancer 的完整伪代码
**指令**:
> "基于已实现的 `redemption_fee.py` 和 `cash_lock.py`，实现 `src/core/engine/rebalancer.py`。
> 核心逻辑:
> 1. `generate_rebalance_plan()` — 输入当前持仓 + 目标权重 + 当前日期 → 输出调仓计划
> 2. 减仓操作: 按 FIFO 顺序遍历 PositionLot → 计算赎回费 → 记录资金解锁时间
> 3. 成本惩罚: 如果 `(新基金预期 Alpha - 旧基金赎回费率) ≤ 0` → 输出 SKIP 标记
> 4. 时间线生成: 输出 T+0 到 T+max_delay 的完整资金解冻-买入时间表
> 5. 输出 `total_friction_cost_yuan` — 所有调仓产生的总摩擦成本"

---

### 切片 5: 数据采集层（30 分钟）

**喂入**: `STRATEGY-AND-DEVELOPMENT.md` §1.2 + 附录A 依赖清单
**指令**:
> "实现 `src/core/data/fetcher.py` 和 `src/core/data/cache.py`。
> - fetcher: 封装 AKShare + 东方财富直连，统一返回 DataFrame
> - cache: SQLite 存静态（基金基本信息、持仓元数据），DuckDB 存时序（日频净值、回测流）
> - 每个表 > 10,000 行必须走 DuckDB
> - 包含自动重试 + 东方财富备用切换逻辑"

---

### 切片 6-15: 后续模块（按 Phase 顺序逐一切片）

每个切片只包含 **一个模块文件** 的完整上下文（对应的 STRATEGY-AND-DEVELOPMENT.md 章节 + RISK-MITIGATION-FRAMEWORK.md 相关代码规范）。

---

## 关键避坑规则（喂给 AI 前的强制检查）

### 规则 1: 视图层零计算检查
```
每次 AI 生成 src/tui/ 或 src/web/ 下的文件后，立即执行:
grep -E "import (pandas|numpy|scipy|sklearn|akshare)" src/tui/ src/web/
grep -E "import (pandas|numpy|scipy|sklearn|akshare)" src/web/
```
任何非 `src.core.api` 的核心库导入 → **无条件打回重写**。

### 规则 2: TODO 零容忍
```
每次 AI 生成代码后，立即执行:
grep -rn "# TODO\|# FIXME\|# HACK\|pass  # " src/
```
任何 TODO 占位符 → **打回，要求补全完整逻辑**。

### 规则 3: asyncio 与 CPU 计算隔离
```
src/tui/ 中涉及回测/验证的函数:
禁止: 直接在 async def 中调用 validate_strategy()
必须: 通过 ProcessPoolExecutor 提交到子进程

src/tui/worker.py 模板:
async def run_backtest():
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, validate_strategy, params)
    return result
```

### 规则 4: VectorBT ≠ Backtrader
```
VectorBT 侧: 不做 CashLockManager，不做赎回费，不做 FIFO
             仅做因子 IC 计算 + 参数粗筛
Backtrader 侧: 完整上线所有中国规则

不可以在 VectorBT 的 @njit 函数中尝试实现资金状态机
```

### 规则 5: FIFO 必须内存化
```
禁止在 SQLite 中做:
  SELECT ... FROM lots WHERE ... ORDER BY purchase_date LIMIT ...
  (性能随持仓数指数爆炸)

必须在内存中:
  lots_deque = deque(sorted(position.lots, key=lambda x: x.purchase_date))
  在 deque 中完成 FIFO 扣减 → 最后批量写入 SQLite
```

---

## 关于本地 LLM 的推荐

**当前可用的最佳组合**:

| 角色 | 模型 | 部署方式 | 适用切片 |
|------|------|---------|---------|
| **主力引擎** | DeepSeek-V4-Pro (OpenCode 内置) | 直接使用 | 所有复杂逻辑切片 |
| **本地备选** | DeepSeek-R1:70b (Ollama) | `ollama pull deepseek-r1:70b` | 切片 3a-c, 6-8 |
| **快速补刀** | Qwen2.5-Coder:32b (Ollama) | `ollama pull qwen2.5-coder:32b` | 切片 2, 5, 模板代码 |

> **OpenCode 中直接使用 DeepSeek-V4-Pro 是最省心的方案**，无需本地部署。只有在需要完全离线或极高吞吐时才用 Ollama。

---

## 执行节奏建议

| 切片 | 预计耗时 | 累积 | 完成后状态 |
|------|---------|------|-----------|
| 0: 契约确认 | 10 min | 10 min | AI 已理解 9 条规则 |
| 1: 目录骨架 | 20 min | 30 min | 目录结构 + 零 UI 泄漏验证通过 |
| 2: 数据 Schema | 30 min | 1 hr | Pydantic Schema + FIFO 验证通过 |
| 3a: 15:00 截点 | 20 min | 1.3 hr | 申赎时间验证器就绪 |
| 3b: 赎回费 | 30 min | 1.8 hr | 阶梯费率计算 + FIFO 持有期就绪 |
| 3c: 资金锁 | 20 min | 2.1 hr | 交收延迟状态机就绪 |
| 4: 智能调仓 | 45 min | 2.9 hr | SmartRebalancer 就绪 |
| 5: 数据层 | 30 min | 3.4 hr | AKShare 封装 + 双数据库就绪 |
| 6+: 策略等 | 逐一切片 | — | 按 Phase 递增 |
