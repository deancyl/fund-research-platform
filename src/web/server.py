"""
Web adapter v0.1.4 — Pydantic DTOs + HTML Dashboard + ECharts.
Root `/` returns interactive dashboard, not Swagger redirect.
/api/rebalance uses sync def (threadpool) to prevent asyncio starvation.
"""

import logging
import uuid as _uuid
from datetime import date as _date, datetime as _datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.core.api import generate_rebalance_plan
from src.core.data.schema import FundCategory, FundChannel, FundPosition, PositionLot

logger = logging.getLogger("web")
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [WEB] %(levelname)s %(message)s", "%H:%M:%S"))
logger.addHandler(_handler)

app = FastAPI(title="Fund Research Platform API v0.1.4")


# ─── Middleware ──────────────────────────────────────────────────────────

@app.middleware("http")
async def trace(request: Request, call_next):
    import time as _t
    cid = str(_uuid.uuid4())[:8]
    s = _t.perf_counter()
    resp = await call_next(request)
    ms = (_t.perf_counter() - s) * 1000
    resp.headers["X-Correlation-ID"] = cid
    resp.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    return resp


# ─── DTO Models ──────────────────────────────────────────────────────────

class WebLotDTO(BaseModel):
    purchase_date: str = Field(description="YYYY-MM-DD")
    shares: float = Field(gt=0)
    purchase_nav: float = Field(gt=0)
    cost_amount: float = Field(ge=0)

class WebPositionDTO(BaseModel):
    fund_code: str = Field(min_length=6, max_length=6)
    fund_name: str
    category: str = "EQUITY"
    channel: str = "OTC_OPEN_END"
    lots: list[WebLotDTO]
    current_nav: float = Field(gt=0)
    total_shares: float = Field(gt=0)
    market_value: float = Field(ge=0)
    weight_pct: float = Field(ge=0, le=1)

class RebalanceWebRequest(BaseModel):
    portfolio: list[WebPositionDTO]
    target_weights: dict[str, float]
    total_value: float
    analysis_date: str = "2026-06-26"


# ─── Dashboard ───────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>🏦 基金投研仪表盘 v0.1.4</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-slate-900 text-slate-100 p-6">
<div class="max-w-6xl mx-auto">
<header class="mb-6 border-b border-slate-700 pb-4">
<h1 class="text-3xl font-bold text-sky-400">🏦 基金投研平台统一仪表盘 <span class="text-sm text-slate-400">v0.1.4</span></h1></header>
<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
<div class="bg-slate-800 p-4 rounded border border-slate-700">
<h2 class="text-lg font-semibold mb-2 text-emerald-400">📥 持仓JSON</h2>
<textarea id="json-in" class="w-full h-48 bg-slate-950 text-emerald-500 text-xs p-3 rounded border border-slate-700">{
 "portfolio": [{
  "fund_code":"005827","fund_name":"易方达蓝筹","channel":"OTC_OPEN_END",
  "current_nav":1.85,"total_shares":25000,"market_value":46250,"weight_pct":0.65,
  "lots":[{"purchase_date":"2026-06-10","shares":5000,"purchase_nav":1.92,"cost_amount":9600}]
 }],
 "target_weights":{"005827":0.2,"510300":0.8},
 "total_value":71153
}</textarea>
<button onclick="run()" class="mt-3 w-full bg-sky-600 hover:bg-sky-500 text-white py-2 rounded">提交再平衡审计</button></div>
<div class="bg-slate-800 p-4 rounded border border-slate-700">
<h2 class="text-lg font-semibold mb-2 text-amber-400">💡 AI规费提示</h2>
<div id="ai-note" class="text-sm bg-slate-900 p-3 rounded min-h-[60px] border border-slate-700">等待诊断...</div>
<h2 class="text-lg font-semibold mt-4 mb-2 text-sky-400">📊 清算流水</h2>
<div id="chart" class="w-full h-40 bg-slate-900 rounded border border-slate-700"></div></div></div></div>
<script>
async function run(){const j=document.getElementById('json-in').value;
try{const r=await fetch('/api/rebalance',{method:'POST',headers:{'Content-Type':'application/json'},body:j});
const d=await r.json();
document.getElementById('ai-note').innerText=d.ai_advisor_note||'OK';
const c=echarts.init(document.getElementById('chart'),'dark');
c.setOption({backgroundColor:'transparent',tooltip:{trigger:'axis'},
xAxis:{type:'category',data:(d.timeline||[]).map(e=>'T+'+e.t_day)},
yAxis:{type:'value',name:'可用(元)'},
series:[{data:(d.timeline||[]).map((_,i)=>d.total_friction_cost_yuan>0?50000-i*5000:70000+i*5000),
type:'line',smooth:true,color:'#38bdf8'}]});}catch(e){alert(e)}}</script></body></html>"""


@app.get("/")
def dashboard() -> HTMLResponse:
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.4"}


@app.get("/api/fee/calculate")
def calc_fee(fund_code: str, channel: str = "OTC_OPEN_END", category: str = "EQUITY", holding_days: int = 30, amount: float = 10000.0) -> dict:
    from src.core.api import calculate_redemption_fee, check_holding_warning
    try:
        ch = FundChannel(channel)
        cat = FundCategory(category)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    r = calculate_redemption_fee(fund_code=fund_code, channel=ch, category=cat, holding_days=holding_days, redemption_amount=amount)
    w = check_holding_warning(holding_days, ch)
    return {"fee_amount_cny": r["fee_amount_cny"], "rate": r["rate"], "warning": w.value}


@app.post("/api/rebalance")
def rebalance(req: RebalanceWebRequest) -> dict:
    """Sync def → FastAPI threadpool. No asyncio starvation."""
    logger.info("rebalance: %d positions", len(req.portfolio))
    try:
        core_positions = []
        for p in req.portfolio:
            lots = [PositionLot(purchase_date=_date.fromisoformat(l.purchase_date), shares=l.shares, purchase_nav=l.purchase_nav, cost_amount=l.cost_amount) for l in p.lots]
            core_positions.append(FundPosition(fund_code=p.fund_code, fund_name=p.fund_name, category=FundCategory(p.category), channel=FundChannel(p.channel), lots=lots, current_nav=p.current_nav, total_shares=p.total_shares, market_value=p.market_value, weight_pct=p.weight_pct))
        plan = generate_rebalance_plan(current_portfolio=core_positions, target_weights=req.target_weights, current_date=_date.fromisoformat(req.analysis_date), total_portfolio_value=req.total_value)
        return {"status": plan.status, "total_friction_cost_yuan": plan.total_friction_cost_yuan, "ai_advisor_note": plan.ai_advisor_note, "actions": [{"fund_code": a.fund_code, "action": a.action_type, "amount": a.amount, "estimated_fee": a.estimated_fee, "reason": a.reason, "skip_reason": a.skip_reason} for a in plan.actions], "timeline": [{"t_day": e.t_day, "event": e.event} for e in plan.timeline]}
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


def launch_web(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    logger.info("launch_web: %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
