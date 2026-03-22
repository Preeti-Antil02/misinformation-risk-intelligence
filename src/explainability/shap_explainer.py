import shap
import numpy as np
import pandas as pd


class XGBExplainer:

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def _to_dense(self, X):
        """Convert sparse matrix to dense if needed."""
        if hasattr(X, "toarray"):
            return X.toarray()
        return X

    def global_importance(self, X):
        X = self._to_dense(X)
        shap_values = self.explainer.shap_values(X)
        importance = np.abs(shap_values).mean(axis=0)

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df

    def explain_instance(self, X, index):
        X = self._to_dense(X)
        shap_values = self.explainer.shap_values(X)
        values = shap_values[index]

        df = pd.DataFrame({
            "feature": self.feature_names,
            "contribution": values
        })

        df["abs"] = df["contribution"].abs()
        df = df.sort_values(by="abs", ascending=False).reset_index(drop=True)

        return df.drop(columns=["abs"])