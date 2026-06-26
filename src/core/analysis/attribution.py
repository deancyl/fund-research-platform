"""
Brinson performance attribution model.

Decomposes active return (portfolio - benchmark) into:
  - Allocation effect: weighting decisions
  - Selection effect: security selection within sectors
  - Interaction effect: cross-product of allocation and selection
"""

import numpy as np


def brinson_attribution(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
    benchmark_total_return: float,
) -> dict[str, float]:
    """
    Brinson-Fachler attribution.

    Returns:
        allocation_effect: Σ(w_p,i - w_b,i) × (R_b,i - R_b)
        selection_effect:  Σ w_b,i × (R_p,i - R_b,i)
        interaction_effect: Σ(w_p,i - w_b,i) × (R_p,i - R_b,i)
        active_return: allocation + selection + interaction
    """
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
        "allocation_effect": round(allocation, 6),
        "selection_effect": round(selection, 6),
        "interaction_effect": round(interaction, 6),
        "active_return": round(allocation + selection + interaction, 6),
    }
