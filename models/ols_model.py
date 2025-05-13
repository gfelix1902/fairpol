import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

class OLSModel:
    def __init__(self, standardize: bool = False):
        self.model = LinearRegression()
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None

    def train(self, X: pd.DataFrame, y: pd.Series):
        X_processed = X.copy()
        if self.standardize and self.scaler is not None:
            X_processed = self.scaler.fit_transform(X_processed)
        self.model.fit(X_processed, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_processed = X.copy()
        if self.standardize and self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
        return pd.Series(self.model.predict(X_processed), index=X.index)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        return {"MSE": mse, "R2": r2}

    def evaluate_conditional_pvalues(self, X_test: pd.DataFrame, y_test: pd.Series, s_test: pd.Series) -> pd.Series:
        group_metrics = {}
        for group in np.unique(s_test):
            mask = s_test == group
            if mask.sum() == 0:
                group_metrics[group] = np.nan
            else:
                y_pred = self.predict(X_test[mask])
                group_metrics[group] = mean_squared_error(y_test[mask], y_pred)
        return pd.Series(group_metrics, name="ols_group_mse")

    def save_model(self, filepath: str):
        joblib.dump({"model": self.model, "scaler": self.scaler, "standardize": self.standardize}, filepath)

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.standardize = data.get("standardize", False)

    def predict_repr(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        return X