"""Spring Festival Effect Strategy (S14) — pre-festival small-cap/ChiNext entry.

Historical evidence (20yr China A-share data):
  - Buying small-cap/ChiNext ETF before Spring Festival and holding for
    20 trading days yields a median return of +9.45% with an 80% win rate.
  - The effect is attributed to pre-festival liquidity injection and
    post-festival retail investor re-entry.

Eligible in ALL market regimes (seasonal, not regime-dependent).
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl  # noqa: TC002
from pydantic import Field

from src.core.strategy.base import BaseStrategy, SignalDirection, StrategyConfig

# ─── Lunar New Year lookup table (2000-2040) ──────────────────────────────────
# Source: Chinese lunar calendar conversion.

_LUNAR_NEW_YEAR: dict[int, date] = {
    2000: date(2000, 2, 5),
    2001: date(2001, 1, 24),
    2002: date(2002, 2, 12),
    2003: date(2003, 2, 1),
    2004: date(2004, 1, 22),
    2005: date(2005, 2, 9),
    2006: date(2006, 1, 29),
    2007: date(2007, 2, 18),
    2008: date(2008, 2, 7),
    2009: date(2009, 1, 26),
    2010: date(2010, 2, 14),
    2011: date(2011, 2, 3),
    2012: date(2012, 1, 23),
    2013: date(2013, 2, 10),
    2014: date(2014, 1, 31),
    2015: date(2015, 2, 19),
    2016: date(2016, 2, 8),
    2017: date(2017, 1, 28),
    2018: date(2018, 2, 16),
    2019: date(2019, 2, 5),
    2020: date(2020, 1, 25),
    2021: date(2021, 2, 12),
    2022: date(2022, 2, 1),
    2023: date(2023, 1, 22),
    2024: date(2024, 2, 10),
    2025: date(2025, 1, 29),
    2026: date(2026, 2, 17),
    2027: date(2027, 2, 6),
    2028: date(2028, 1, 26),
    2029: date(2029, 2, 13),
    2030: date(2030, 2, 3),
    2031: date(2031, 1, 23),
    2032: date(2032, 2, 11),
    2033: date(2033, 1, 31),
    2034: date(2034, 2, 19),
    2035: date(2035, 2, 8),
    2036: date(2036, 1, 28),
    2037: date(2037, 2, 15),
    2038: date(2038, 2, 4),
    2039: date(2039, 1, 24),
    2040: date(2040, 2, 12),
}

_REQUIRED_FIELDS: list[str] = ["close"]


# ─── Configuration ────────────────────────────────────────────────────────────


class SpringFestivalConfig(StrategyConfig):
    """Configuration for Spring Festival Effect strategy."""

    hold_days: int = Field(
        default=20,
        ge=1,
        description="Number of trading days to hold after purchase",
    )
    pre_days: int = Field(
        default=5,
        ge=1,
        description="Number of calendar days before Spring Festival to enter",
    )
    target_fund: str = Field(
        default="159915",
        description="Target ETF code (default: ChiNext ETF 159915)",
    )


# ─── Strategy ─────────────────────────────────────────────────────────────────


class SpringFestival(BaseStrategy):
    """Spring Festival Effect strategy.

    Generates a BUY signal when the current date falls within pre_days
    calendar days before the nearest upcoming Spring Festival (Lunar New Year).
    The confidence scales with proximity to the festival date.
    """

    config: SpringFestivalConfig = Field(
        default_factory=lambda: SpringFestivalConfig(name="spring_festival"),
        description="Spring Festival strategy configuration",
    )

    def generate_signals(
        self,
        dt: date,
        market_data: dict[str, pl.DataFrame],
        portfolio: object = None,  # noqa: ARG002
    ) -> list[dict[str, float | int | str]]:
        """Generate BUY signal if within pre-festival window.

        Args:
            dt: Trading date.
            market_data: Must contain the target_fund's DataFrame.
            portfolio: Current portfolio (unused).

        Returns:
            List with a single BUY signal, or empty list.

        """
        target = self.config.target_fund

        if target not in market_data or market_data[target].is_empty():
            return []

        if not self.is_in_pre_festival_window(dt):
            return []

        # Confidence: closer to festival = higher confidence (0.5 to 0.95)
        sf = self.get_spring_festival(dt.year)
        if sf is None:
            return []

        days_until = (sf - dt).days
        max_offset = self.config.pre_days
        confidence = 0.95 - (0.45 * (days_until / max_offset))

        return [{
            "fund_code": target,
            "direction": SignalDirection.BUY.value,
            "confidence": round(confidence, 4),
            "target_weight": self.config.max_position_pct,
            "reason": f"Spring Festival effect: {days_until}d until Lunar New Year ({sf})",
        }]

    def required_data(self) -> list[str]:
        """Declare required data fields."""
        return _REQUIRED_FIELDS

    def get_spring_festival(self, year: int) -> date | None:
        """Return the Spring Festival (Lunar New Year) date for a given year.

        Args:
            year: Gregorian year.

        Returns:
            The lunar new year date, or None if outside lookup range.

        """
        return _LUNAR_NEW_YEAR.get(year)

    def is_in_pre_festival_window(self, dt: date) -> bool:
        """Check if a date falls within the pre-festival purchase window.

        The window is [festival - pre_days, festival), i.e., pre_days
        calendar days before the Spring Festival up to (but not including)
        the festival day itself.

        Args:
            dt: Date to check.

        Returns:
            True if dt is within the pre-festival window.

        """
        sf = self.get_spring_festival(dt.year)
        if sf is None:
            return False

        window_start = sf - timedelta(days=self.config.pre_days)
        return window_start <= dt < sf
