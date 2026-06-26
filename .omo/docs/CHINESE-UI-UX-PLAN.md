# 中文 UI/UX 交互改进计划

**版本**: v0.3.2 | **优先级**: 高

---

## 一、现状诊断

### TUI 终端
| 问题 | 现状 | 严重度 |
|------|------|--------|
| 列标题为英文 | `Code, Name, NAV, Change%` | 🔴 |
| 状态栏混用中英文 | `r=持仓诊断 s=策略轮动 q=退出` | 🟡 |
| 日志中文编码 | 部分日志乱码(Windows GBK/UTF-8) | 🔴 |
| 快捷键提示 | Footer 默认英文 | 🟡 |
| 输入占位符 | 中文但无 CJK 字体适配 | 🟡 |

### Web 终端
| 问题 | 现状 | 严重度 |
|------|------|--------|
| 组件标签英文 | `Buy, Sell, Amount, Fee` 等 | 🔴 |
| 审计按钮 | `Execute Audit` 而非 `发起审计` | 🟡 |
| 风控状态 | `● Engine Ready` 而非 `● 引擎就绪` | 🟡 |
| 滑块提示 | `%` 符号无千分位格式化 | 🟡 |

---

## 二、改进计划（按优先级）

### Phase 1: TUI 全中文界面（本周）

1. **列标题中文化**:
   - `Code→代码, Name→名称, NAV→净值, Change%→涨跌%`
   - 推荐面板: `Strategy→策略, Signal→信号, Confidence→置信度, Reason→理由`

2. **状态栏统一中文**:
   - 所有 `self._update_status()` 和 `notify()` 消息使用纯中文

3. **日志编码修复**:
   - 日志 handler 显式设置 `encoding='utf-8'`
   - 或使用 `rich.logging.RichHandler` (支持 CJK)

4. **快捷键中文显示**:
   ```python
   BINDINGS = [
       ("q", "quit", "退出"),
       ("r", "refresh", "刷新数据"),
       ("s", "run_strategies", "运行策略"),
       ("f5", "market_data", "行情总览"),
       ("f9", "deep_analysis", "深度资料"),
       ("l", "toggle_log", "日志开关"),
   ]
   ```

### Phase 2: Web 全中文界面（本周）

1. **组件标签中文化**:
   - 表头: `代码/名称/通道/净值/权重`
   - 决策面板: `赎回/申购/金额/摩擦成本`
   - 风控提示: `⏳ 清算中... / ⚡ 发起再平衡审计`

2. **滑块区域中文**:
   - `目标权重 (拖拽滑块)` → `🎚️ 目标权重`
   - 归一化按钮: `Normalize → 归一化`
   - 状态: `等待注入持仓数据...`

3. **Agent 辩论流中文**:
   ```javascript
   addLog('🧠 Agent辩论系统已连接');
   addLog('📊 宏观策略师: 分析宏观指标...');
   addLog('📈 量化分析师: 计算因子暴露...');
   addLog('🛡️ 风控官: 检查持仓集中度...');
   addLog('🎯 CIO: 综合判决输出...');
   ```

### Phase 3: 字体与国际化基础设施（下周）

1. **CJK 字体适配**: 引入 `Source Han Sans` 或系统默认中文字体
2. **数字格式化**: 金额使用千分位 `¥1,234,567`
3. **国际化框架**: 预留 `i18n.py` 模块，key-value 映射

---

## 三、具体代码改动清单

| 文件 | 改动 |
|------|------|
| `src/tui/app.py` | BINDINGS 中文, 列标题中文, 状态栏中文, RichHandler |
| `src/web/server.py` | DASHBOARD 表头/按钮/状态全中文 |
| `src/cli/main.py` | CLI help 文本中文 |
| `src/core/i18n.py` | **新建** — 中英文 key-value 映射表 |

---

## 四、验收标准

- [ ] TUI 启动界面所有可见文本为简体中文
- [ ] Web 仪表盘所有组件标签/按钮/提示为简体中文
- [ ] 日志无乱码 (Windows GBK 兼容)
- [ ] 710 tests GREEN
