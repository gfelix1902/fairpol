import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import joblib
import logging

class OLSModel:
    def __init__(self, standardize: bool = False, feature_selection: bool = False, threshold: float = 0.01, cv: int = None, regularization: str = None, alpha: float = 1.0, impute: bool = False):
        if regularization == "ridge":
            self.model = Ridge(alpha=alpha, n_jobs=-1)
        elif regularization == "lasso":
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
        self.standardize = standardize
        self.feature_selection = feature_selection
        self.threshold = threshold
        self.scaler = StandardScaler() if standardize else None
        self.selector = VarianceThreshold(threshold=threshold) if feature_selection else None
        self.cv = cv
        self.impute = impute
        self.imputer = SimpleImputer(strategy="mean") if impute else None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.logger.info("Starting training...")
        X_processed = X.copy()
        
        # Interaktionsterme hinzufügen
      
        if "trainy1" in X_processed.columns and "trainy2" in X_processed.columns:
            X_processed["trainy1_x_trainy2"] = X_processed["trainy1"] * X_processed["trainy2"]
                
        # Speichere die Feature-Namen für spätere Verwendung
        self.feature_names_in_ = X_processed.columns.tolist()
        
        if self.impute and self.imputer is not None:
            X_processed = self.imputer.fit_transform(X_processed)
        if self.feature_selection and self.selector is not None:
            X_processed = self.selector.fit_transform(X_processed)
        if self.standardize and self.scaler is not None:
            X_processed = self.scaler.fit_transform(X_processed)  # <--- Hier wird der Scaler angepasst
        if self.cv:
            scores = cross_val_score(self.model, X_processed, y, cv=self.cv, scoring="neg_mean_squared_error")
            print(f"Cross-Validation MSE: {-scores.mean():.4f} ± {scores.std():.4f}")
        self.model.fit(X_processed, y)
        self.feature_names_in_ = X.columns.tolist()  # <--- Hier werden die Spaltennamen gespeichert
        self.logger.info("Training completed.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_processed = X.copy()
        if self.impute and self.imputer is not None:
            X_processed = self.imputer.transform(X_processed)
        if self.feature_selection and self.selector is not None:
            X_processed = self.selector.transform(X_processed)
        if self.standardize and self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)  # <--- Hier wird der Scaler angewendet
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

    def evaluate_conditional_metrics(self, X_test: pd.DataFrame, y_test: pd.Series, s_test: pd.Series) -> pd.DataFrame:
        group_metrics = []
        for group in np.unique(s_test):
            mask = s_test == group
            if mask.sum() == 0:
                group_metrics.append({"group": group, "MSE": np.nan, "R2": np.nan, "MAE": np.nan})
            else:
                y_pred = self.predict(X_test[mask])
                mse = mean_squared_error(y_test[mask], y_pred)
                r2 = r2_score(y_test[mask], y_pred)
                mae = mean_absolute_error(y_test[mask], y_pred)
                group_metrics.append({"group": group, "MSE": mse, "R2": r2, "MAE": mae})
        return pd.DataFrame(group_metrics)

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

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        if hasattr(self.model, "coef_"):
            importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": self.model.coef_
            }).sort_values(by="Importance", ascending=False)
            return importance
        else:
            raise ValueError("Model does not have coefficients.")

    def predict_ite(self, X: pd.DataFrame, treat_col: str = "assignment") -> np.ndarray:
        X1 = X.copy()
        X1[treat_col] = 1
        X0 = X.copy()
        X0[treat_col] = 0
        # Stelle sicher, dass die Reihenfolge der Spalten wie beim Training ist!
        if hasattr(self, "feature_order") and self.feature_order is not None:
            X1 = X1[self.feature_order]
            X0 = X0[self.feature_order]
        y1_pred = self.predict(X1)
        y0_pred = self.predict(X0)
        return (y1_pred - y0_pred).values
    
    def predict_cate(self, X: pd.DataFrame, treat_cols: list, treat_values: list, base_values: list = None) -> np.ndarray:
        X_treat = X.copy()
        X_base = X.copy()
        
        # Setze Treatment-Werte
        for col, val in zip(treat_cols, treat_values):
            X_treat[col] = val
            
        if base_values is None:
            base_values = [0] * len(treat_cols)
            
        for col, val in zip(treat_cols, base_values):
            X_base[col] = val
            
        # Interaktionsterme hinzufügen
        if "trainy1" in X_treat.columns and "trainy2" in X_treat.columns:
            X_treat["trainy1_x_trainy2"] = X_treat["trainy1"] * X_treat["trainy2"]
            X_base["trainy1_x_trainy2"] = X_base["trainy1"] * X_base["trainy2"]
        
        # Weitere Verarbeitung wie bisher...
        if hasattr(self, "feature_order") and self.feature_order is not None:
            X_treat = X_treat[self.feature_order]
            X_base = X_base[self.feature_order]
            
        y_treat = self.predict(X_treat)
        y_base = self.predict(X_base)
        
        return (y_treat - y_base).values

    def add_interactions(df):
        # Kopie erstellen, um das Original nicht zu verändern
        df_new = df.copy()
        
        # Interaktionsterm für Training in beiden Jahren
        df_new["trainy1_x_trainy2"] = df_new["trainy1"] * df_new["trainy2"]
        
        # Weitere Interaktionen könnten hinzugefügt werden, z.B.:
        # df_new["age_x_educ"] = df_new["age"] * df_new["educ"]
        
        return df_new