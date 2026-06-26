"""
Web adapter — FastAPI REST + WebSocket. v0.1.3
Per audit: Pydantic models, sync def rebalance (threadpool), root→/docs redirect.
"""

import logging
import time as _time
import uuid as _uuid
from datetime import date as _date, datetime as _datetime

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse

from src.core.api import generate_rebalance_plan, calculate_redemption_fee, check_holding_warning
from src.core.data.schema import FundChannel, FundCategory, FundPosition, PositionLot

logger = logging.getLogger("web")
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [WEB] %(levelname)s %(message)s", "%H:%M:%S"))
logger.addHandler(_handler)

app = FastAPI(title="Fund Research Platform API", version="0.1.3")


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    cid = str(_uuid.uuid4())[:8]
    s = _time.perf_counter()
    logger.info("REQ %s %s [%s]", request.method, request.url.path, cid)
    resp = await call_next(request)
    ms = (_time.perf_counter() - s) * 1000
    resp.headers["X-Correlation-ID"] = cid
    resp.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    logger.info("RES %s → %d (%.1fms) [%s]", request.url.path, resp.status_code, ms, cid)
    return resp


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.3", "timestamp": _datetime.now().isoformat()}


@app.get("/api/fee/calculate")
async def calc_fee(fund_code: str, channel: str = "OTC_OPEN_END", category: str = "EQUITY", holding_days: int = 30, amount: float = 10000.0) -> dict:
    try:
        ch = FundChannel(channel)
        cat = FundCategory(category)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    r = calculate_redemption_fee(fund_code=fund_code, channel=ch, category=cat, holding_days=holding_days, redemption_amount=amount)
    w = check_holding_warning(holding_days, ch)
    return {"fee_amount_cny": r["fee_amount_cny"], "rate": r["rate"], "warning": w.value, "holding_days": holding_days, "computed_at": _datetime.now().isoformat()}


@app.post("/api/rebalance")
def rebalance(payload: dict) -> dict:
    """Sync def → FastAPI threadpool. No asyncio starvation."""
    logger.info("rebalance: %d pos, %d targets", len(payload.get("portfolio", [])), len(payload.get("target_weights", {})))
    try:
        portfolio = []
        for pd in payload.get("portfolio", []):
            lots = [PositionLot(purchase_date=_date.fromisoformat(l["purchase_date"]), shares=float(l["shares"]), purchase_nav=float(l["purchase_nav"]), cost_amount=float(l["cost_amount"])) for l in pd.get("lots", [])]
            portfolio.append(FundPosition(fund_code=pd["fund_code"], fund_name=pd.get("fund_name", pd["fund_code"]), category=FundCategory(pd.get("category", "EQUITY")), channel=FundChannel(pd.get("channel", "OTC_OPEN_END")), lots=lots, current_nav=float(pd["current_nav"]), total_shares=float(pd["total_shares"]), market_value=float(pd["market_value"]), weight_pct=float(pd["weight_pct"])))
        plan = generate_rebalance_plan(current_portfolio=portfolio, target_weights=payload["target_weights"], current_date=_date.fromisoformat(payload.get("date", str(_date.today()))), total_portfolio_value=float(payload["total_value"]))
        return {"status": plan.status, "actions": [{"type": a.action_type, "fund_code": a.fund_code, "amount": a.amount, "estimated_fee": a.estimated_fee, "net_cash": a.net_cash, "reason": a.reason, "skip_reason": a.skip_reason} for a in plan.actions], "total_friction_cost_yuan": plan.total_friction_cost_yuan, "ai_advisor_note": plan.ai_advisor_note, "computed_at": _datetime.now().isoformat()}
    except (KeyError, ValueError, TypeError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    cid = str(_uuid.uuid4())[:6]
    logger.info("WS: %s", cid)
    await ws.send_json({"type": "connected", "client_id": cid})
    try:
        while True:
            d = await ws.receive_text()
            await ws.send_json({"type": "ack", "received": d[:200]})
    except WebSocketDisconnect:
        logger.info("WS dc: %s", cid)
    except Exception:
        logger.exception("WS err: %s", cid)


def launch_web(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    logger.info("launch_web: %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
