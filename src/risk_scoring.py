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
            return "High"
        else:
            return "Critical"

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