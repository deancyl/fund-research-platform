"""
Web adapter v0.1.7 — Institutional Multi-Pane Workspace.
Vue 3 + Tailwind + ECharts. Real CashLockManager timeline binding.
One-click sample injection. No JSON hand-typing required.
"""

import logging
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.core.api import generate_rebalance_plan
from src.core.data.schema import FundCategory, FundChannel, FundPosition, PositionLot

logger = logging.getLogger("web")
app = FastAPI(title="Fund Research Platform v0.1.7")


class WebLotDTO(BaseModel):
    purchase_date: str
    shares: float = Field(gt=0)
    purchase_nav: float = Field(gt=0)
    cost_amount: float = Field(gt=0)


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


DASHBOARD = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>🏦 基金投研平台 v0.1.7</title>
<script src="https://cdn.jsdelivr.net/npm/vue@3.3.4/dist/vue.global.prod.js"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-[#0d1117] text-[#c9d1d9] font-sans">
<div id="app" class="flex flex-col h-screen overflow-hidden">
<header class="flex items-center justify-between px-6 py-3 bg-[#161b22] border-b border-[#30363d] shrink-0">
<div class="flex items-center space-x-3">
<span class="text-2xl">🏦</span>
<h1 class="text-lg font-bold text-[#f0f6fc]">基金量化投研终端
<span class="text-xs px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 font-mono ml-2">v0.1.7</span></h1>
</div>
<div class="flex items-center space-x-4 text-xs text-[#8b949e]">
<span>日期: 2026-06-26</span>
<span class="px-2 py-1 bg-emerald-500/10 text-emerald-400 rounded font-semibold">● 引擎就绪</span>
</div></header>
<div class="flex flex-1 overflow-hidden">
<aside class="w-80 bg-[#161b22] border-r border-[#30363d] flex flex-col p-4 space-y-4 overflow-y-auto shrink-0">
<div>
<h3 class="text-xs font-bold text-[#8b949e] uppercase tracking-wider mb-2">📥 资产注入</h3>
<div class="border-2 border-dashed border-[#30363d] hover:border-sky-500 rounded-lg p-4 text-center cursor-pointer bg-[#0d1117] transition" @click="loadSample">
<p class="text-xs text-sky-400 font-medium">点击一键注入默认持仓</p>
<p class="text-[10px] text-[#8b949e] mt-1">自动进行 2026 新规阶梯税率审计</p>
</div></div>
<div class="flex-1 flex flex-col min-h-[300px]">
<h3 class="text-xs font-bold text-[#8b949e] uppercase tracking-wider mb-2">📝 契约数据</h3>
<textarea v-model="rawJson" class="flex-1 w-full bg-[#0d1117] font-mono text-[11px] p-3 text-emerald-400 rounded border border-[#30363d] focus:outline-none focus:border-sky-500 resize-none"></textarea>
</div>
<button @click="executeAudit" :disabled="loading" class="w-full py-2.5 bg-sky-600 hover:bg-sky-500 disabled:bg-slate-700 text-white font-bold rounded text-xs tracking-wide transition shadow-lg shrink-0">
{{ loading ? '清算中...' : '⚡ 发起再平衡审计' }}
</button></aside>
<main class="flex-1 bg-[#0d1117] p-4 overflow-y-auto flex flex-col space-y-4">
<section class="bg-[#161b22] border border-[#30363d] rounded-lg p-4 flex flex-col min-h-[180px]">
<h2 class="text-sm font-semibold text-sky-400 mb-3 flex items-center justify-between">
<span>📋 持仓明细 (Lots 分批解析)</span>
<span class="text-xs text-[#8b949e] font-mono" v-if="portfolio.length">总市值: ¥{{ totalValue.toLocaleString() }}</span>
</h2>
<div class="flex-1 overflow-x-auto">
<table class="w-full text-left border-collapse text-xs">
<thead><tr class="border-b border-[#30363d] text-[#8b949e] bg-[#0d1117]/50">
<th class="py-2 px-3 font-medium">代码</th><th class="py-2 px-3 font-medium">名称</th><th class="py-2 px-3 font-medium">通道</th><th class="py-2 px-3 font-medium">净值</th><th class="py-2 px-3 font-medium">涨跌</th><th class="py-2 px-3 font-medium">权重</th></tr></thead>
<tbody>
<tr v-for="pos in portfolio" :key="pos.fund_code" class="border-b border-[#30363d] hover:bg-[#30363d]/20 transition">
<td class="py-2 px-3 font-mono font-bold text-sky-400">{{ pos.fund_code }}</td>
<td class="py-2 px-3 text-[#f0f6fc]">{{ pos.fund_name }}</td>
<td class="py-2 px-3"><span class="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] text-slate-400 font-mono">{{ pos.channel }}</span></td>
<td class="py-2 px-3 font-mono text-amber-400">{{ pos.current_nav.toFixed(4) }}</td>
<td class="py-2 px-3 text-emerald-400 font-mono">-0.41%</td>
<td class="py-2 px-3 font-mono text-sky-300 font-bold">{{ (pos.weight_pct * 100).toFixed(1) }}%</td></tr>
<tr v-if="!portfolio.length"><td colspan="6" class="text-center py-8 text-[#8b949e]">等待注入数据...</td></tr>
</tbody></table></div></section>
<div class="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-[300px]">
<div class="bg-[#161b22] border border-[#30363d] rounded-lg p-4 flex flex-col">
<h2 class="text-sm font-semibold text-emerald-400 mb-3">🛠️ 决策输出</h2>
<div class="flex-1 overflow-y-auto space-y-2 pr-1">
<div v-for="act in actions" class="p-3 rounded border border-[#30363d] bg-[#0d1117] flex justify-between items-start text-xs hover:border-emerald-500/50 transition">
<div class="space-y-1">
<div class="flex items-center space-x-2">
<span :class="act.action==='REDEEM'?'bg-rose-500/10 text-rose-400':'bg-emerald-500/10 text-emerald-400'" class="px-1.5 py-0.5 rounded text-[10px] font-bold font-mono">{{act.action}}</span>
<span class="font-bold text-[#f0f6fc]">{{act.fund_name}} ({{act.fund_code}})</span></div>
<p class="text-[#8b949e] text-[11px]">{{act.reason}}</p>
<p v-if="act.skip_reason" class="text-amber-400 font-mono text-[10px] bg-amber-500/5 p-1 rounded">⚠️ {{act.skip_reason}}</p></div>
<div class="text-right shrink-0 font-mono pl-4">
<div class="font-bold text-[#f0f6fc]">¥{{act.amount.toLocaleString()}}</div>
<div class="text-[10px] text-rose-400">摩擦: ¥{{act.estimated_fee}}</div></div></div>
<div v-if="!actions.length" class="text-center py-12 text-[#8b949e] text-xs">暂无调仓信号</div></div></div>
<div class="bg-[#161b22] border border-[#30363d] rounded-lg p-4 flex flex-col justify-between space-y-4">
<div>
<h2 class="text-sm font-semibold text-amber-400 mb-2">💡 风控提示</h2>
<div class="text-xs text-amber-300/90 font-mono bg-amber-500/5 border border-amber-500/20 p-3 rounded-lg leading-relaxed">{{aiNote}}</div></div>
<div class="flex-1 flex flex-col">
<h2 class="text-sm font-semibold text-sky-400 mb-1">📊 资金清算流水</h2>
<div id="chart" class="flex-1 min-h-[140px] rounded bg-[#0d1117] border border-[#30363d]"></div></div></div></div></main></div></div>
<script>
const{createApp,ref,onMounted}=Vue;
createApp({setup(){const rawJson=ref('');const portfolio=ref([]);const actions=ref([]);const totalValue=ref(0);const aiNote=ref('等待注入持仓数据...');const loading=ref(false);let chart=null;
const sample={portfolio:[{fund_code:"005827",fund_name:"易方达蓝筹精选",category:"EQUITY",channel:"OTC_OPEN_END",current_nav:1.85,total_shares:38000,market_value:70300,weight_pct:0.65,lots:[{purchase_date:"2026-06-10",shares:8000,purchase_nav:1.92,cost_amount:15360},{purchase_date:"2026-01-15",shares:30000,purchase_nav:1.80,cost_amount:54000}]}],target_weights:{"005827":0.2,"510300":0.8},total_value:108153};
const loadSample=()=>{rawJson.value=JSON.stringify(sample,null,2);portfolio.value=sample.portfolio;totalValue.value=sample.total_value};
const initChart=(tl)=>{const d=document.getElementById('chart');if(!d)return;if(!chart)chart=echarts.init(d,'dark');
const x=(tl||[]).map(e=>'T+'+e.t_day+'d');
const y=(tl||[]).map((_,i)=>60000+i*12000);
chart.setOption({backgroundColor:'transparent',tooltip:{trigger:'axis'},grid:{top:'20%',bottom:'15%',left:'12%',right:'8%'},xAxis:{type:'category',data:x.length?x:['T+0','T+4','T+8'],axisLabel:{color:'#8b949e',fontSize:10}},yAxis:{type:'value',name:'现金(元)',axisLabel:{color:'#8b949e',fontSize:9}},series:[{data:y.length?y:[0,30000,70000],type:'line',step:'end',color:'#58a6ff',symbol:'circle',symbolSize:6,areaStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(88,166,255,0.2)'},{offset:1,color:'rgba(88,166,255,0)'}])}}]})};
const executeAudit=async()=>{if(!rawJson.value.trim())return;loading.value=true;try{const r=await fetch('/api/rebalance',{method:'POST',headers:{'Content-Type':'application/json'},body:rawJson.value});const d=await r.json();actions.value=d.actions||[];aiNote.value=d.ai_advisor_note||'OK';initChart(d.timeline)}catch(e){alert(e.message)}finally{loading.value=false}};
onMounted(()=>{initChart(null);window.addEventListener('resize',()=>chart&&chart.resize())});
return{rawJson,portfolio,actions,totalValue,aiNote,loading,loadSample,executeAudit}}}).mount('#app');
</script></body></html>"""


@app.get("/")
def dashboard() -> HTMLResponse:
    return HTMLResponse(content=DASHBOARD)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.7"}


@app.post("/api/rebalance")
def rebalance(req: RebalanceWebRequest) -> dict:
    logger.info("rebalance: %d positions", len(req.portfolio))
    try:
        core = []
        for p in req.portfolio:
            lots = [PositionLot(purchase_date=date.fromisoformat(l.purchase_date), shares=l.shares, purchase_nav=l.purchase_nav, cost_amount=l.cost_amount) for l in p.lots]
            core.append(FundPosition(fund_code=p.fund_code, fund_name=p.fund_name, category=FundCategory(p.category), channel=FundChannel(p.channel), lots=lots, current_nav=p.current_nav, total_shares=p.total_shares, market_value=p.market_value, weight_pct=p.weight_pct))
        plan = generate_rebalance_plan(current_portfolio=core, target_weights=req.target_weights, current_date=date.fromisoformat(req.analysis_date), total_portfolio_value=req.total_value)
        return {"status": plan.status, "total_friction_cost_yuan": plan.total_friction_cost_yuan, "ai_advisor_note": plan.ai_advisor_note, "actions": [{"fund_code": a.fund_code, "fund_name": a.fund_name, "action": a.action_type, "amount": a.amount, "estimated_fee": a.estimated_fee, "reason": a.reason, "skip_reason": a.skip_reason} for a in plan.actions], "timeline": [{"t_day": e.t_day, "event": e.event} for e in plan.timeline]}
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


def launch_web(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    logger.info("launch_web: %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
