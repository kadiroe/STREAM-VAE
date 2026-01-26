from __future__ import annotations

class BaseDetector:
    """
    Minimal base class (TSB-AD-like) used by StreamVAE.
    """
    def __init__(self):
        self.__anomaly_score = None

    def fit(self, data):
        raise NotImplementedError

    def decision_function(self, data):
        raise NotImplementedError

    def anomaly_score(self):
        return self.__anomaly_score

    def _set_anomaly_score(self, s):
        self.__anomaly_score = s
