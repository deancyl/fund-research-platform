"""
HMM Market Regime Detection (Phase 5A stub).
In production, install hmmlearn for full GaussianHMM.
This stub provides the 3-state regime classification interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RegimeProbabilities:
    trending_up: float
    trending_down: float
    sideways: float
    dominant_regime: str


class HMMRegimeDetector:
    """
    3-state Hidden Markov Model for market regime detection.

    Features: return_1d, return_5d, volatility_20d, rsi_14, volume_ratio (from config.yaml).
    States: trending_up, trending_down, sideways.

    In production: replace _predict_stub() with hmmlearn's GaussianHMM.predict().
    """

    N_STATES: int = 3

    def __init__(self) -> None:
        pass

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit HMM and return state sequence.

        Args:
            features: (n, 5) array of [ret_1d, ret_5d, vol_20d, rsi_14, vol_ratio].

        Returns:
            (n,) array of state labels: 0=trending_up, 1=down, 2=sideways.
        """
        # Try hmmlearn, fallback to heuristic
        try:
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(n_components=self.N_STATES, covariance_type="full", n_iter=100, random_state=42)
            model.fit(features)
            return model.predict(features)
        except ImportError:
            return self._heuristic_predict(features)

    def classify_current(self, features: np.ndarray) -> RegimeProbabilities:
        """
        Classify the most recent data point into regime probabilities.
        Uses latest 60 observations as context.
        """
        n = len(features)
        if n < 20:
            return RegimeProbabilities(0.33, 0.33, 0.34, "sideways")

        window = features[-60:] if n >= 60 else features
        states = self.fit_predict(window)
        latest_state = states[-1]

        # Count state frequencies for pseudo-probabilities
        unique, counts = np.unique(states, return_counts=True)
        probs = dict(zip(unique, counts / len(states)))

        return RegimeProbabilities(
            trending_up=float(probs.get(0, 0.0)),
            trending_down=float(probs.get(1, 0.0)),
            sideways=float(probs.get(2, 0.0)),
            dominant_regime={0: "trending_up", 1: "trending_down", 2: "sideways"}.get(latest_state, "sideways"),
        )

    @staticmethod
    def _heuristic_predict(features: np.ndarray) -> np.ndarray:
        """Heuristic fallback: use ret_1d and vol_20d to classify."""
        n = len(features)
        states = np.zeros(n, dtype=int)  # default sideways

        for i in range(n):
            ret_1d = features[i, 0] if features.shape[1] > 0 else 0
            vol_20d = features[i, 2] if features.shape[1] > 2 else 0.01
            rsi = features[i, 3] if features.shape[1] > 3 else 50

            if ret_1d > vol_20d and rsi > 55:
                states[i] = 0  # trending_up
            elif ret_1d < -vol_20d and rsi < 45:
                states[i] = 1  # trending_down
            else:
                states[i] = 2  # sideways

        return states
