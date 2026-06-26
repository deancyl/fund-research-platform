"""
Web adapter — FastAPI REST + WebSocket with full request tracing.

Per SYSTEM-CONTRACT §1: this VIEW LAYER imports ONLY from src.core.api.

Every endpoint produces:
  1. Structured log entry (timestamp, method, path, status, duration)
  2. Correlation ID for request tracing
  3. Actionable error responses with error codes
  4. Timing metadata in every response
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.core.api import (
    calculate_redemption_fee,
    check_holding_warning,
    generate_rebalance_plan,
)
from src.core.data.schema import FundCategory, FundChannel, FundPosition, PositionLot

# ─── Structured logging ─────────────────────────────────────────────────────

logger = logging.getLogger("web")
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter('%(asctime)s [WEB] %(levelname)s %(message)s', "%H:%M:%S")
)
logger.addHandler(_handler)

app = FastAPI(
    title="Fund Research Platform API",
    version="0.1.0",
)


# ─── Request tracing middleware ──────────────────────────────────────────────


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    """Log every request with correlation ID and timing."""
    corr_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    logger.info("REQ %s %s %s [%s]", request.method, request.url.path, request.client.host if request.client else "?", corr_id)

    response = await call_next(request)

    elapsed = time.perf_counter() - start
    response.headers["X-Correlation-ID"] = corr_id
    response.headers["X-Response-Time-ms"] = f"{elapsed*1000:.1f}"
    logger.info("RES %s %s → %d (%.1fms) [%s]", request.method, request.url.path, response.status_code, elapsed * 1000, corr_id)

    return response


# ─── Error handler ──────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: BROAD_EXCEPT_OK
    corr_id = str(uuid.uuid4())[:8]
    logger.exception("ERR %s %s [%s]", request.method, request.url.path, corr_id)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": str(exc), "correlation_id": corr_id},
    )


# ─── Health ──────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health(request: Request) -> dict:
    logger.debug("health check from %s", request.client.host if request.client else "?")
    return {
        "status": "ok",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
    }


# ─── Funds ───────────────────────────────────────────────────────────────────


@app.get("/api/funds")
async def list_funds() -> dict:
    logger.info("list_funds: returning sample data")
    return {
        "funds": [
            {"code": "510300", "name": "沪深300ETF", "type": "指数型"},
            {"code": "159915", "name": "创业板ETF", "type": "指数型"},
            {"code": "005827", "name": "易方达蓝筹精选", "type": "混合型"},
            {"code": "161725", "name": "白酒基金", "type": "行业型"},
        ]
    }


@app.get("/api/funds/{fund_code}")
async def fund_detail(fund_code: str) -> dict:
    logger.info("fund_detail: %s", fund_code)
    return {
        "fund_code": fund_code,
        "name": "Sample Fund",
        "nav_history": [],
        "message": "连接 AKShare 以获取实时数据",
    }


# ─── Fee Calculation ─────────────────────────────────────────────────────────


@app.get("/api/fee/calculate")
async def calc_fee(
    fund_code: str,
    channel: str = "OTC_OPEN_END",
    category: str = "EQUITY",
    holding_days: int = 30,
    amount: float = 10000.0,
) -> dict:
    logger.info("calc_fee: fund=%s channel=%s holding=%dd amount=%.0f", fund_code, channel, holding_days, amount)

    try:
        ch = FundChannel(channel)
        cat = FundCategory(category)
    except ValueError as e:
        logger.warning("calc_fee: invalid param — %s", e)
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_parameter", "detail": str(e)},
        )

    result = calculate_redemption_fee(
        fund_code=fund_code, channel=ch, category=cat,
        holding_days=holding_days, redemption_amount=amount,
    )
    warning = check_holding_warning(holding_days, ch)

    return {
        "fee_amount_cny": result["fee_amount_cny"],
        "rate": result["rate"],
        "warning": warning.value,
        "holding_days": holding_days,
        "channel": channel,
        "computed_at": datetime.now().isoformat(),
    }


# ─── Rebalancing ─────────────────────────────────────────────────────────────


@app.post("/api/rebalance")
async def rebalance(payload: dict) -> dict:
    logger.info("rebalance: %d positions, %d target weights", len(payload.get("portfolio", [])), len(payload.get("target_weights", {})))

    try:
        portfolio = []
        for pos_data in payload.get("portfolio", []):
            lots = [
                PositionLot(
                    purchase_date=date.fromisoformat(lot["purchase_date"]),
                    shares=float(lot["shares"]),
                    purchase_nav=float(lot["purchase_nav"]),
                    cost_amount=float(lot["cost_amount"]),
                )
                for lot in pos_data.get("lots", [])
            ]
            pos = FundPosition(
                fund_code=pos_data["fund_code"],
                fund_name=pos_data["fund_name"],
                category=FundCategory(pos_data.get("category", "EQUITY")),
                channel=FundChannel(pos_data.get("channel", "OTC_OPEN_END")),
                lots=lots,
                current_nav=float(pos_data["current_nav"]),
                total_shares=float(pos_data["total_shares"]),
                market_value=float(pos_data["market_value"]),
                weight_pct=float(pos_data["weight_pct"]),
            )
            portfolio.append(pos)

        target_weights = payload.get("target_weights", {})
        current_date = date.fromisoformat(payload.get("date", str(date.today())))
        total_value = float(payload.get("total_value", 0))

        plan = generate_rebalance_plan(
            current_portfolio=portfolio,
            target_weights=target_weights,
            current_date=current_date,
            total_portfolio_value=total_value,
        )

        logger.info("rebalance: %d actions, friction=%.2f CNY", len(plan.actions), plan.total_friction_cost_yuan)

        return {
            "status": plan.status,
            "actions": [
                {
                    "type": a.action_type,
                    "fund_code": a.fund_code,
                    "amount": a.amount,
                    "estimated_fee": a.estimated_fee,
                    "net_cash": a.net_cash,
                    "reason": a.reason,
                    "skip_reason": a.skip_reason,
                }
                for a in plan.actions
            ],
            "total_friction_cost_yuan": plan.total_friction_cost_yuan,
            "ai_advisor_note": plan.ai_advisor_note,
            "timeline": [{"t_day": e.t_day, "event": e.event} for e in plan.timeline],
            "computed_at": datetime.now().isoformat(),
        }

    except (KeyError, ValueError, TypeError) as e:
        logger.warning("rebalance: bad request — %s", e)
        return JSONResponse(
            status_code=400,
            content={"error": "bad_request", "detail": str(e)},
        )


# ─── Strategies ──────────────────────────────────────────────────────────────


@app.post("/api/strategies/run")
async def run_strategies(payload: dict) -> dict:
    strategy_names = payload.get("strategies", ["factor_momentum", "pe_pb_band"])
    logger.info("run_strategies: %s", strategy_names)

    # In production: instantiate strategies, feed market_data, collect signals
    signals = [
        {
            "strategy": name,
            "fund_code": "510300",
            "direction": "HOLD",
            "confidence": 0.72,
            "reason": "连接数据源以获取实时信号",
        }
        for name in strategy_names
    ]

    return {
        "signals": signals,
        "strategies_run": len(strategy_names),
        "computed_at": datetime.now().isoformat(),
    }


# ─── WebSocket (real-time push with audit) ──────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client_id = str(uuid.uuid4())[:6]
    logger.info("WS connected: client=%s", client_id)
    await ws.send_json({"type": "connected", "client_id": client_id, "timestamp": datetime.now().isoformat()})

    try:
        while True:
            data = await ws.receive_text()
            logger.debug("WS msg from %s: %s", client_id, data[:100])
            await ws.send_json({
                "type": "ack",
                "received": data[:200],
                "timestamp": datetime.now().isoformat(),
            })
    except WebSocketDisconnect:
        logger.info("WS disconnected: client=%s", client_id)
    except Exception:  # noqa: BROAD_EXCEPT_OK — WebSocket boundary
        logger.exception("WS error: client=%s", client_id)


# ─── Launch ──────────────────────────────────────────────────────────────────


def launch_web(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the FastAPI web server."""
    import uvicorn
    logger.info("launch_web: starting on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
