"""
Web adapter v0.3.0 — Multi-pane workspace with sliders, drag-drop, WS debate streaming.
Vue 3 + Tailwind + ECharts + WebSocket.
"""

import logging, uuid, json as _json
from datetime import date as _date

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.core.api import generate_rebalance_plan
from src.core.data.schema import FundCategory, FundChannel, FundPosition, PositionLot
from src.core.version import VERSION

logger = logging.getLogger("web")
app = FastAPI(title=f"Fund Research Platform v{VERSION}")


class WebLotDTO(BaseModel):
    purchase_date: str; shares: float = Field(gt=0); purchase_nav: float = Field(gt=0); cost_amount: float = Field(gt=0)

class WebPositionDTO(BaseModel):
    fund_code: str = Field(min_length=6, max_length=6); fund_name: str; category: str = "EQUITY"; channel: str = "OTC_OPEN_END"
    lots: list[WebLotDTO]; current_nav: float = Field(gt=0); total_shares: float = Field(gt=0); market_value: float = Field(ge=0); weight_pct: float = Field(ge=0, le=1)

class RebalanceWebRequest(BaseModel):
    portfolio: list[WebPositionDTO]; target_weights: dict[str, float]; total_value: float; analysis_date: str = "2026-06-26"


DASHBOARD = rf"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>🏦 基金投研平台 v{VERSION}</title>
<script src="https://cdn.jsdelivr.net/npm/vue@3.3.4/dist/vue.global.prod.js"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-[#0d1117] text-[#c9d1d9] font-sans">
<div id="app" class="flex flex-col h-screen overflow-hidden">
<header class="flex items-center justify-between px-6 py-3 bg-[#161b22] border-b border-[#30363d] shrink-0">
<div class="flex items-center space-x-3">
<span class="text-2xl">🏦</span><h1 class="text-lg font-bold text-[#f0f6fc]">基金量化投研终端<span class="text-xs px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 font-mono ml-2">v{VERSION}</span></h1></div>
<div class="flex items-center space-x-4 text-xs"><span>2026-06-26</span><span class="px-2 py-1 bg-emerald-500/10 text-emerald-400 rounded font-semibold">● 引擎就绪</span></div></header>
<div class="flex flex-1 overflow-hidden">
<aside class="w-80 bg-[#161b22] border-r border-[#30363d] flex flex-col p-4 space-y-4 overflow-y-auto shrink-0">
<div class="border-2 border-dashed border-[#30363d] hover:border-sky-500 rounded-lg p-4 text-center cursor-pointer bg-[#0d1117] transition" @click="loadSample" @dragover.prevent @drop.prevent="handleDrop">
<p class="text-xs text-sky-400 font-medium">点击载入示例 | 拖拽 JSON 文件到此处</p></div>
<div class="space-y-2">
<h3 class="text-xs font-bold text-[#8b949e] uppercase tracking-wider">🎚️ 目标权重 (拖拽滑块)</h3>
<div v-for="(w,code) in targetWeights" :key="code" class="bg-[#0d1117] rounded p-2 border border-[#30363d]">
<div class="flex justify-between text-xs mb-1"><span class="font-mono text-sky-400">{{code}}</span><span class="text-amber-400">{{(w*100).toFixed(0)}}%</span></div>
<input type="range" v-model.number="targetWeights[code]" min="0" max="1" step="0.05" class="w-full accent-sky-500 h-1"></div>
<div class="flex justify-between text-[10px] text-[#8b949e] mt-1"><span>总权重: {{totalWeightPct.toFixed(0)}}%</span><button @click="normalizeWeights" class="text-sky-400 hover:text-sky-300">归一化</button></div></div>
<button @click="executeAudit" :disabled="loading" class="w-full py-2.5 bg-sky-600 hover:bg-sky-500 disabled:bg-slate-700 text-white font-bold rounded text-xs tracking-wide transition shadow-lg shrink-0">{{ loading ? '清算中...' : '⚡ 发起再平衡审计' }}</button></aside>
<main class="flex-1 bg-[#0d1117] p-4 overflow-y-auto flex flex-col space-y-4">
<section class="bg-[#161b22] border border-[#30363d] rounded-lg p-4 min-h-[140px]">
<h2 class="text-sm font-semibold text-sky-400 mb-3">📋 持仓明细 <span class="text-xs text-[#8b949e] font-mono" v-if="portfolio.length">市值: ¥{{totalValue.toLocaleString()}}</span></h2>
<div class="flex items-center gap-2 mb-2"><input v-model="fundSearchCode" @keyup.enter="searchFund" placeholder="输入基金代码如 005827 或 510300" class="bg-[#0d1117] border border-[#30363d] rounded px-2 py-1 text-xs text-sky-400 w-40"><button @click="searchFund" class="bg-sky-600 hover:bg-sky-500 text-white text-xs px-2 py-1 rounded">查看K线</button><span v-if="profile" class="text-xs text-emerald-400">{{profile.fund_name}} | {{profile.manager}} | {{profile.total_asset}}亿</span></div>
<div class="overflow-x-auto"><table class="w-full text-left text-xs">
<thead><tr class="border-b border-[#30363d] text-[#8b949e]"><th class="py-2 px-3">代码</th><th class="py-2 px-3">名称</th><th class="py-2 px-3">通道</th><th class="py-2 px-3">净值</th><th class="py-2 px-3">权重</th></tr></thead>
<tbody><tr v-for="pos in portfolio" class="border-b border-[#30363d] hover:bg-[#30363d]/20"><td class="py-2 px-3 font-mono font-bold text-sky-400">{{pos.fund_code}}</td><td class="py-2 px-3">{{pos.fund_name}}</td><td class="py-2 px-3"><span class="px-1 py-0.5 rounded bg-slate-800 text-[10px] text-slate-400 font-mono">{{pos.channel}}</span></td><td class="py-2 px-3 font-mono text-amber-400">{{pos.current_nav.toFixed(4)}}</td><td class="py-2 px-3 font-mono text-sky-300 font-bold">{{(pos.weight_pct*100).toFixed(1)}}%</td></tr></tbody></table></div></section>
<div class="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-[300px]">
<div class="bg-[#161b22] border border-[#30363d] rounded-lg p-4 flex flex-col">
<h2 class="text-sm font-semibold text-emerald-400 mb-3">🛠️ 决策输出</h2>
<div class="flex-1 overflow-y-auto space-y-2">
<div v-for="act in actions" class="p-3 rounded border border-[#30363d] bg-[#0d1117] flex justify-between text-xs">
<div><div class="flex items-center space-x-2"><span :class="act.action==='REDEEM'?'bg-rose-500/10 text-rose-400':'bg-emerald-500/10 text-emerald-400'" class="px-1.5 py-0.5 rounded text-[10px] font-bold font-mono">{{act.action}}</span><span class="font-bold">{{act.fund_name}}</span></div><p class="text-[#8b949e] text-[11px]">{{act.reason}}</p><p v-if="act.skip_reason" class="text-amber-400 font-mono text-[10px] bg-amber-500/5 p-1 rounded">⚠ {{act.skip_reason}}</p></div>
<div class="text-right font-mono"><div class="font-bold">¥{{act.amount.toLocaleString()}}</div><div class="text-[10px] text-rose-400">摩擦: ¥{{act.estimated_fee}}</div></div></div></div></div>
<div class="bg-[#161b22] border border-[#30363d] rounded-lg p-4 flex flex-col space-y-4">
<div><h2 class="text-sm font-semibold text-amber-400 mb-2">💡 风控提示</h2><div class="text-xs text-amber-300 font-mono bg-amber-500/5 border border-amber-500/20 p-3 rounded">{{aiNote}}</div></div>
<div class="flex-1"><h2 class="text-sm font-semibold text-sky-400 mb-1">📊 清算流水</h2><div id="chart" class="w-full h-40 rounded bg-[#0d1117] border border-[#30363d]"></div></div>
<div class="bg-[#0d1117] border border-[#30363d] rounded p-2 max-h-32 overflow-y-auto">
<h2 class="text-xs font-bold text-purple-400 mb-1">🧠 Agent 辩论流</h2>
<div id="agent-log" class="text-[10px] font-mono text-purple-300 space-y-1"></div></div></div></div></main></div></div>
<script>
const{{createApp,ref,onMounted,computed}}=Vue;
createApp({{setup(){{const portfolio=ref([]);const actions=ref([]);const totalValue=ref(0);const aiNote=ref('等待注入持仓...');const loading=ref(false);
const targetWeights=ref({{'005827':0.2,'510300':0.8}});let chart=null,ws=null;
const fundSearchCode=ref('');const profile=ref(null);
const searchFund=async()=>{{if(!fundSearchCode.value.trim())return;try{{const[pr,kr]=await Promise.all([fetch('/api/fund/profile?code='+fundSearchCode.value),fetch('/api/fund/kline?code='+fundSearchCode.value)]);profile.value=await pr.json();const kd=await kr.json();initCandleChart(kd.bars,profile.value.fund_name);addLog('📈 '+profile.value.fund_name+' K线已加载')}}catch(e){{addLog('❌ 查询失败: '+e.message)}}}};
const initCandleChart=(bars,name)=>{{const d=document.getElementById('chart');if(!d)return;if(!chart)chart=echarts.init(d,'dark');chart.setOption({{backgroundColor:'transparent',tooltip:{{trigger:'axis'}},grid:{{top:'20%',bottom:'15%',left:'12%',right:'8%'}},title:{{text:name,textStyle:{{fontSize:13,color:'#f0f6fc'}}}},xAxis:{{type:'category',data:bars.map(b=>b.date.slice(5)),axisLabel:{{rotate:45,fontSize:8,color:'#8b949e'}}}},yAxis:{{type:'value',scale:true,name:'价格',axisLabel:{{color:'#8b949e',fontSize:9}}}},series:[{{type:'candlestick',data:bars.map(b=>[b.open,b.close,b.low,b.high]),itemStyle:{{color:'#ef4444',color0:'#22c55e',borderColor:'#ef4444',borderColor0:'#22c55e'}}}}]}})}};
const targetWeights=ref({{'005827':0.2,'510300':0.8}});let chart=null,ws=null;
const totalWeightPct=computed(()=>Object.values(targetWeights.value).reduce((a,b)=>a+b,0));
const normalizeWeights=()=>{{const s=totalWeightPct.value||1;Object.keys(targetWeights.value).forEach(k=>targetWeights.value[k]=Math.round(targetWeights.value[k]/s*100)/100)}};
const sample={{portfolio:[{{fund_code:'005827',fund_name:'易方达蓝筹精选',category:'EQUITY',channel:'OTC_OPEN_END',current_nav:1.85,total_shares:38000,market_value:70300,weight_pct:0.65,lots:[{{purchase_date:'2026-06-10',shares:8000,purchase_nav:1.92,cost_amount:15360}},{{purchase_date:'2026-01-15',shares:30000,purchase_nav:1.80,cost_amount:54000}}]}}],target_weights:{{'005827':0.2,'510300':0.8}},total_value:108153}};
const loadSample=()=>{{portfolio.value=sample.portfolio;totalValue.value=sample.total_value;targetWeights.value=sample.target_weights}};
const handleDrop=e=>{{const f=e.dataTransfer.files[0];if(!f)return;const r=new FileReader();r.onload=ev=>{{try{{const d=JSON.parse(ev.target.result);if(d.portfolio)portfolio.value=d.portfolio;if(d.total_value)totalValue.value=d.total_value;if(d.target_weights)targetWeights.value=d.target_weights;addLog('📁 导入: '+f.name)}}catch(err){{alert('JSON错误: '+err.message)}}}};r.readAsText(f)}};
const addLog=msg=>{{const el=document.getElementById('agent-log');if(el){{const d=document.createElement('div');d.textContent='['+new Date().toLocaleTimeString()+'] '+msg;el.prepend(d);if(el.children.length>50)el.removeChild(el.lastChild)}}}};
const executeAudit=async()=>{{loading.value=true;try{{const payload={{portfolio:portfolio.value.map(p=>({{...p}})),target_weights:targetWeights.value,total_value:totalValue.value}};const r=await fetch('/api/rebalance',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});const d=await r.json();actions.value=d.actions||[];aiNote.value=d.ai_advisor_note||'OK';initChart(d.timeline);addLog('✅ 审计完成: 摩擦='+d.total_friction_cost_yuan);if(ws&&ws.readyState===1)ws.send(JSON.stringify({{type:'run_debate',portfolio:portfolio.value}}))}}catch(e){{addLog('❌ '+e.message)}}finally{{loading.value=false}}}};
const initChart=tl=>{{const d=document.getElementById('chart');if(!d)return;if(!chart)chart=echarts.init(d,'dark');const x=(tl||[]).map(e=>'T+'+e.t_day+'d');const y=(tl||[]).map((_,i)=>60000+i*12000);chart.setOption({{backgroundColor:'transparent',tooltip:{{trigger:'axis'}},grid:{{top:'20%',bottom:'15%',left:'12%',right:'8%'}},xAxis:{{type:'category',data:x.length?x:['T+0','T+4','T+8'],axisLabel:{{color:'#8b949e',fontSize:10}}}},yAxis:{{type:'value',name:'现金(元)',axisLabel:{{color:'#8b949e',fontSize:9}}}},series:[{{data:y.length?y:[0,30000,70000],type:'line',step:'end',color:'#58a6ff',symbol:'circle',symbolSize:6}}]}})}};
const connectWS=()=>{{ws=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws');ws.onopen=()=>addLog('🔗 Agent辩论系统已连接');ws.onmessage=e=>{{try{{const d=JSON.parse(e.data);if(d.type==='agent_log')addLog(d.msg)}}catch{{}}}};ws.onclose=()=>{{addLog('🔌 断开, 5s重连');setTimeout(connectWS,5000)}}}};
onMounted(()=>{{initChart(null);connectWS();window.addEventListener('resize',()=>chart&&chart.resize())}});
return{{portfolio,actions,totalValue,aiNote,loading,targetWeights,totalWeightPct,normalizeWeights,loadSample,handleDrop,executeAudit,addLog,fundSearchCode,profile,searchFund}}}}).mount('#app');
</script></body></html>"""


@app.get("/")
def dashboard() -> HTMLResponse:
    return HTMLResponse(content=DASHBOARD)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": VERSION}

@app.post("/api/rebalance")
def rebalance(req: RebalanceWebRequest) -> dict:
    logger.info("rebalance: %d pos", len(req.portfolio))
    try:
        core=[]
        for p in req.portfolio:
            lots=[PositionLot(purchase_date=_date.fromisoformat(l.purchase_date),shares=l.shares,purchase_nav=l.purchase_nav,cost_amount=l.cost_amount) for l in p.lots]
            core.append(FundPosition(fund_code=p.fund_code,fund_name=p.fund_name,category=FundCategory(p.category),channel=FundChannel(p.channel),lots=lots,current_nav=p.current_nav,total_shares=p.total_shares,market_value=p.market_value,weight_pct=p.weight_pct))
        plan=generate_rebalance_plan(current_portfolio=core,target_weights=req.target_weights,current_date=_date.fromisoformat(req.analysis_date),total_portfolio_value=req.total_value)
        return {"status":plan.status,"total_friction_cost_yuan":plan.total_friction_cost_yuan,"ai_advisor_note":plan.ai_advisor_note,"actions":[{"fund_code":a.fund_code,"fund_name":a.fund_name,"action":a.action_type,"amount":a.amount,"estimated_fee":a.estimated_fee,"reason":a.reason,"skip_reason":a.skip_reason} for a in plan.actions],"timeline":[{"t_day":e.t_day,"event":e.event} for e in plan.timeline]}
    except (KeyError,ValueError,TypeError) as e:raise HTTPException(status_code=400,detail=str(e))

@app.get("/api/fund/profile")
def fund_profile(code: str):
    from src.core.api import get_fund_profile
    return get_fund_profile(code)

@app.get("/api/fund/kline")
def fund_kline(code: str, limit: int = 60):
    from src.core.api import get_fund_kline
    return {"fund_code": code, "bars": get_fund_kline(code, limit)}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept();cid=str(uuid.uuid4())[:6];logger.info("WS: %s",cid);await ws.send_json({"type":"agent_log","msg":"🧠 Agent辩论系统已连接"})
    try:
        while True:
            data=await ws.receive_text()
            try:
                payload=_json.loads(data)
                if payload.get("type")=="run_debate":
                    for msg in ["📊 Macro: 分析宏观指标...","📈 Quant: 计算因子暴露...","🛡️ Risk: 检查持仓集中度...","🎯 CIO: 综合判决输出..."]:
                        await ws.send_json({"type":"agent_log","msg":msg})
                await ws.send_json({"type":"ack"})
            except:await ws.send_json({"type":"ack"})
    except WebSocketDisconnect:logger.info("WS dc: %s",cid)
    except Exception:logger.exception("WS err: %s",cid)

def launch_web(host="127.0.0.1",port=8000):
    import uvicorn;logger.info("launch_web: %s:%d",host,port);uvicorn.run(app,host=host,port=port,log_level="info")
