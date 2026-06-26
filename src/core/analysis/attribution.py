"""
Brinson performance attribution model. v0.1.9: multi-period with Carino smoothing.

Single-period Brinson-Fachler:
  Allocation:  Σ(w_p - w_b) × (R_b - R_benchmark)
  Selection:   Σ w_b × (R_p - R_b)
  Interaction: Σ(w_p - w_b) × (R_p - R_b)

Multi-period Carino (per audit §2B):
  k_t = [ln(1+R_p) - ln(1+R_b)] / [R_p - R_b]
  Multi-period effect = Σ(k_t × effect_t) / Σ(k_t) — additive after smoothing.
"""

import numpy as np


def brinson_attribution(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
    benchmark_total_return: float,
) -> dict[str, float]:
    allocation = 0.0
    selection = 0.0
    interaction = 0.0
    all_sectors = set(portfolio_weights) | set(benchmark_weights)
    for sector in all_sectors:
        wp = portfolio_weights.get(sector, 0.0)
        wb = benchmark_weights.get(sector, 0.0)
        rp = portfolio_returns.get(sector, 0.0)
        rb = benchmark_returns.get(sector, 0.0)
        allocation += (wp - wb) * (rb - benchmark_total_return)
        selection += wb * (rp - rb)
        interaction += (wp - wb) * (rp - rb)
    return {
        "allocation_effect": round(allocation, 6), "selection_effect": round(selection, 6),
        "interaction_effect": round(interaction, 6), "active_return": round(allocation + selection + interaction, 6),
    }


def _portfolio_return(weights: dict, returns: dict) -> float:
    return sum(weights.get(s, 0.0) * returns.get(s, 0.0) for s in set(weights) | set(returns))


def multi_period_brinson(periods: list[dict]) -> dict[str, float]:
    n = len(periods)
    if n == 0:
        return {"allocation_effect": 0.0, "selection_effect": 0.0, "interaction_effect": 0.0, "active_return": 0.0, "periods": 0}
    if n == 1:
        p = periods[0]
        r = brinson_attribution(p["portfolio_weights"], p["benchmark_weights"], p["portfolio_returns"], p["benchmark_returns"], p["benchmark_total_return"])
        r["periods"] = 1
        return r

    k_sum = 0.0
    aw, sw, iw = 0.0, 0.0, 0.0
    cum_excess, cum_bench = 0.0, 0.0

    for p in periods:
        pw, bw = p["portfolio_weights"], p["benchmark_weights"]
        pr, br = p["portfolio_returns"], p["benchmark_returns"]
        rb = p["benchmark_total_return"]
        rp = p.get("portfolio_total_return", _portfolio_return(pw, pr))

        k_t = (np.log(1.0 + rp) - np.log(1.0 + rb)) / (rp - rb) if abs(rp - rb) > 1e-10 else 1.0 / (1.0 + rp)
        sp = brinson_attribution(pw, bw, pr, br, rb)
        k_sum += k_t
        aw += k_t * sp["allocation_effect"]
        sw += k_t * sp["selection_effect"]
        iw += k_t * sp["interaction_effect"]
        cum_excess = (1.0 + cum_excess) * (1.0 + rp - rb) - 1.0
        cum_bench = (1.0 + cum_bench) * (1.0 + rb) - 1.0

    return {
        "allocation_effect": round(aw / max(k_sum, 1e-12), 6), "selection_effect": round(sw / max(k_sum, 1e-12), 6),
        "interaction_effect": round(iw / max(k_sum, 1e-12), 6), "active_return": round(cum_excess, 6),
        "cumulative_benchmark": round(cum_bench, 6), "periods": n,
    }
