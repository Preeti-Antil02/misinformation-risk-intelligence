<<<<<<< HEAD
"""
src/risk_scoring.py

Converts continuous model probability into standardized, well-calibrated risk tiers.
Updated per V2.0 Specification:
- Low:      p < 0.30
- Moderate: 0.30 <= p < 0.60
- High:     0.60 <= p < 0.85
- Critical: p >= 0.85
"""

from typing import List, Union
import numpy as np


class RiskScorer:
    """
    Categorizes misinformation risk probabilities into standardized tiers.
    """

    def __init__(self):
        # Precise V2.0 Thresholds
        self.thresholds = {
            "low": 0.30,
            "moderate": 0.60,
            "high": 0.85
        }
        self.ensemble_thresholds = {
            "low": 0.30,
            "moderate": 0.60,
            "high": 0.85
        }

    def get_risk_level(self, prob: float) -> str:
        """
        Maps a base model probability to a standard risk level.
        """
        p = float(prob)
        if p < self.thresholds["low"]:
            return "Low"
        elif p < self.thresholds["moderate"]:
            return "Moderate"
        elif p < self.thresholds["high"]:
=======
# src/risk_scoring.py

class RiskScorer:
    """
    Converts model probability into human-readable risk levels
    """

    def __init__(self):
        self.thresholds = {
            "low": 0.3,
            "moderate": 0.6,
            "high": 0.85
        }

    def get_risk_level(self, prob):
        if prob < self.thresholds["low"]:
            return "Low"
        elif prob < self.thresholds["moderate"]:
            return "Moderate"
        elif prob < self.thresholds["high"]:
>>>>>>> origin/main
            return "High"
        else:
            return "Critical"

<<<<<<< HEAD
    def score_batch(self, probabilities: Union[List[float], np.ndarray]) -> List[str]:
        """
        Scores an iterable of continuous probabilities.
        """
        return [self.get_risk_level(p) for p in probabilities]

    def score(self, prob: float) -> str:
        """
        Alias for get_risk_level.
        """
        return self.get_risk_level(prob)

    def score_ensemble(self, prob: float) -> str:
        """
        Maps a calibrated ensemble probability into calibrated risk tiers.
        """
        p = float(prob)
        if p < self.ensemble_thresholds["low"]:
            return "Low"
        elif p < self.ensemble_thresholds["moderate"]:
            return "Moderate"
        elif p < self.ensemble_thresholds["high"]:
            return "High"
        else:
            return "Critical"
=======
    def score_batch(self, probabilities):
        return [self.get_risk_level(p) for p in probabilities]
    
    def score(self, prob):
        return self.get_risk_level(prob)
    
    def score_ensemble(self, prob):
        if prob < 0.25:
            return "Low"
        elif prob < 0.50:
            return "Moderate"
        elif prob < 0.70:
            return "High"
        else:
            return "Critical"
>>>>>>> origin/main
