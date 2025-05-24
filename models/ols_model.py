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
import os # Für os.makedirs

class OLSModel:
    def __init__(self, 
                 standardize: bool = False, 
                 feature_selection: bool = False, 
                 threshold: float = 0.01, 
                 cv: int = None, 
                 regularization: str = None, 
                 alpha: float = 1.0, 
                 impute: bool = False,
                 interaction_covariates: list = None): # Neuer Parameter
        
        self.regularization_type = regularization # Für Persistenz speichern
        self.alpha_value = alpha # Für Persistenz speichern
        if regularization == "ridge":
            self.model = Ridge(alpha=alpha) # n_jobs=-1 entfernt, da nicht Standard für Ridge/Lasso
        elif regularization == "lasso":
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
            
        self.standardize = standardize
        self.feature_selection = feature_selection
        self.threshold = threshold
        self.scaler = StandardScaler() if standardize else None
        self.selector = VarianceThreshold(threshold=threshold) if feature_selection else None
        self.cv = cv # Wird für cross_val_score verwendet, nicht direkt im Modell gespeichert
        self.impute = impute
        self.imputer = SimpleImputer(strategy="mean") if impute else None
        
        self.interaction_covariates = interaction_covariates if interaction_covariates else []
        self.feature_names_in_ = None # Wird im Training gesetzt

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _add_interaction_terms(self, X_df: pd.DataFrame) -> pd.DataFrame:
        X_processed = X_df.copy()
        if "trainy1" in X_processed.columns and "trainy2" in X_processed.columns:
            X_processed["trainy1_x_trainy2"] = X_processed["trainy1"] * X_processed["trainy2"]
        # Füge "assignment" zur Liste der Treatment-Variablen hinzu
        treatment_columns_for_interactions = ["assignment", "trainy1", "trainy2", "trainy1_x_trainy2"]
        for treat_col in treatment_columns_for_interactions:
            if treat_col not in X_processed.columns:
                continue
            for cov_col in self.interaction_covariates:
                if cov_col in X_processed.columns:
                    interaction_col_name = f"{treat_col}_x_{cov_col}"
                    X_processed[interaction_col_name] = X_processed[treat_col] * X_processed[cov_col]
        return X_processed

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.logger.info("Starting training...")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X to OLSModel.train must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("Input y to OLSModel.train must be a pandas Series.")

        X_processed = X.copy()
        
        # 1. Interaktionsterme hinzufügen (vor Imputation, Selektion, Skalierung)
        X_processed = self._add_interaction_terms(X_processed)
        
        current_feature_names = X_processed.columns.tolist()
        
        # 2. Imputation
        if self.impute and self.imputer is not None:
            X_imputed_array = self.imputer.fit_transform(X_processed)
            X_processed = pd.DataFrame(X_imputed_array, columns=current_feature_names, index=X_processed.index)
            # current_feature_names bleiben gleich

        # 3. Feature Selektion
        if self.feature_selection and self.selector is not None:
            self.selector.fit(X_processed) 
            selected_features_mask = self.selector.get_support()
            X_processed = X_processed.loc[:, selected_features_mask]
            current_feature_names = X_processed.columns.tolist() # Namen nach Selektion aktualisieren

        # 4. Standardisierung
        if self.standardize and self.scaler is not None:
            X_scaled_array = self.scaler.fit_transform(X_processed)
            X_processed = pd.DataFrame(X_scaled_array, columns=current_feature_names, index=X_processed.index)
            # current_feature_names bleiben gleich

        self.feature_names_in_ = X_processed.columns.tolist() # Finale Feature-Namen speichern
        
        if self.cv is not None and self.cv > 1:
            try:
                scores = cross_val_score(self.model, X_processed, y, cv=self.cv, scoring="neg_mean_squared_error")
                self.logger.info(f"Cross-Validation MSE: {-scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                self.logger.error(f"Error during cross-validation: {e}")
        
        self.model.fit(X_processed, y) 
        self.logger.info("Training completed.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not isinstance(X, pd.DataFrame):
            self.logger.error("Input X to OLSModel.predict must be a pandas DataFrame.")
            raise TypeError("Input X to OLSModel.predict must be a pandas DataFrame.")
        if self.feature_names_in_ is None:
            self.logger.error("Model has not been trained yet or feature_names_in_ not set. Call train() first.")
            raise ValueError("Model not trained or feature_names_in_ missing.")

        X_predict = X.copy()
        original_index = X_predict.index
        
        # 1. Interaktionsterme hinzufügen (konsistent zum Training)
        X_predict = self._add_interaction_terms(X_predict)
        
        current_cols_for_transform = X_predict.columns.tolist()

        # 2. Imputation
        if self.impute and self.imputer is not None:
            imputed_data = self.imputer.transform(X_predict)
            X_predict = pd.DataFrame(imputed_data, columns=current_cols_for_transform, index=original_index)

        # 3. Feature Selektion
        if self.feature_selection and self.selector is not None:
            selected_data = self.selector.transform(X_predict)
            
            cols_after_selection = self.feature_names_in_ if not self.standardize else [f for f in self.feature_names_in_ if f in X_predict.columns[self.selector.get_support()]]
            X_predict = pd.DataFrame(selected_data, columns=self.feature_names_in_ if not self.standardize else X_predict.columns[self.selector.get_support()], index=original_index)
            if isinstance(selected_data, np.ndarray):
                 X_predict = pd.DataFrame(selected_data, columns=self.feature_names_in_, index=original_index)


        # 4. Standardisierung
        if self.standardize and self.scaler is not None:
            scaled_data = self.scaler.transform(X_predict)
            X_predict = pd.DataFrame(scaled_data, columns=self.feature_names_in_, index=original_index)

        X_for_model_predict = X_predict[self.feature_names_in_]

        return pd.Series(self.model.predict(X_for_model_predict), index=original_index)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        return {"MSE": mse, "R2": r2, "MAE": mae}

    def evaluate_conditional_metrics(self, X_test: pd.DataFrame, y_test: pd.Series, s_test: pd.Series) -> pd.DataFrame:
        group_metrics_list = []
        unique_groups = np.unique(s_test) if s_test is not None else [None] # Handle case where s_test might be None

        for group in unique_groups:
            if group is None: # Evaluate on all data if no groups
                mask = pd.Series([True] * len(X_test), index=X_test.index)
            else:
                mask = (s_test == group)
            
            if mask.sum() == 0:
                group_metrics_list.append({"group": group, "MSE": np.nan, "R2": np.nan, "MAE": np.nan, "count": 0})
            else:
                X_group = X_test[mask]
                y_group = y_test[mask]
                if X_group.empty:
                    group_metrics_list.append({"group": group, "MSE": np.nan, "R2": np.nan, "MAE": np.nan, "count": 0})
                    continue
                
                y_pred_group = self.predict(X_group)
                mse = mean_squared_error(y_group, y_pred_group)
                r2 = r2_score(y_group, y_pred_group)
                mae = mean_absolute_error(y_group, y_pred_group)
                group_metrics_list.append({"group": group, "MSE": mse, "R2": r2, "MAE": mae, "count": len(X_group)})
        return pd.DataFrame(group_metrics_list)

    def save_model(self, filepath: str):
        # Sicherstellen, dass der Ordner existiert
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data_to_save = {
            "model_params": self.model.get_params(), # Scikit-learn Modellparameter
            "model_coef": getattr(self.model, "coef_", None),
            "model_intercept": getattr(self.model, "intercept_", None),
            "model_class": self.model.__class__, # Zum Wiederherstellen des richtigen Modelltyps
            "scaler": self.scaler,
            "standardize": self.standardize,
            "selector": self.selector,
            "feature_selection": self.feature_selection,
            "threshold": self.threshold,
            "imputer": self.imputer,
            "impute": self.impute,
            "feature_names_in_": self.feature_names_in_,
            "interaction_covariates": self.interaction_covariates,
            "cv_param": self.cv,
            "regularization_type": self.regularization_type,
            "alpha_value": self.alpha_value
        }
        joblib.dump(data_to_save, filepath)
        self.logger.info(f"Modell gespeichert unter: {filepath}")

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        
        # Modell wiederherstellen
        model_class = data["model_class"]
        self.model = model_class(**data.get("model_params", {})) # Parameter übergeben
        if data.get("model_coef") is not None:
            self.model.coef_ = data["model_coef"]
        if data.get("model_intercept") is not None:
            self.model.intercept_ = data["model_intercept"]
        if hasattr(self.model, "coef_") and self.model.coef_ is not None:
             # Markiert das Modell als gefittet für einige sklearn Prüfungen
            if not hasattr(self.model, "n_features_in_") and data.get("feature_names_in_"):
                self.model.n_features_in_ = len(data["feature_names_in_"])


        self.scaler = data.get("scaler")
        self.standardize = data.get("standardize", False)
        self.selector = data.get("selector")
        self.feature_selection = data.get("feature_selection", False)
        self.threshold = data.get("threshold", 0.01)
        self.imputer = data.get("imputer")
        self.impute = data.get("impute", False)
        self.feature_names_in_ = data.get("feature_names_in_")
        self.interaction_covariates = data.get("interaction_covariates", [])
        self.cv = data.get("cv_param")
        self.regularization_type = data.get("regularization_type")
        self.alpha_value = data.get("alpha_value")
        
        self.logger.info(f"Modell geladen von: {filepath}")
        if self.feature_names_in_ is None:
            self.logger.warning("Geladenes Modell hat keine 'feature_names_in_'.")
        if not (hasattr(self.model, "coef_") and self.model.coef_ is not None):
             self.logger.warning("Koeffizienten des geladenen Modells nicht gesetzt. Modell ist möglicherweise nicht voll funktionsfähig.")


    def get_feature_importance(self) -> pd.DataFrame: # feature_names nicht mehr als Argument benötigt
        if hasattr(self.model, "coef_") and self.feature_names_in_ is not None:
            if len(self.feature_names_in_) == len(self.model.coef_):
                importance = pd.DataFrame({
                    "Feature": self.feature_names_in_,
                    "Importance": self.model.coef_
                }).sort_values(by="Importance", key=abs, ascending=False) # Sort by absolute importance
                return importance
            else:
                self.logger.error(f"Länge von feature_names_in_ ({len(self.feature_names_in_)}) stimmt nicht mit Anzahl der Koeffizienten ({len(self.model.coef_)}) überein.")
                return pd.DataFrame() # Empty DataFrame on error
        else:
            self.logger.warning("Modell hat keine Koeffizienten oder feature_names_in_ ist nicht gesetzt.")
            return pd.DataFrame()

    def predict_ite(self, X: pd.DataFrame, treat_col: str = "assignment") -> np.ndarray:
        X1 = X.copy()
        X1[treat_col] = 1
        X0 = X.copy()
        X0[treat_col] = 0
        
        # Die predict-Methode kümmert sich um das Hinzufügen aller Interaktionsterme
        y1_pred = self.predict(X1)
        y0_pred = self.predict(X0)
        return (y1_pred - y0_pred).values
    
    def predict_cate(self, X: pd.DataFrame, treat_cols: list, treat_values: list, base_values: list = None) -> np.ndarray:
        X_treat = X.copy()
        X_base = X.copy()
        
        for col, val in zip(treat_cols, treat_values):
            if col in X_treat.columns:
                X_treat[col] = val
            else:
                self.logger.warning(f"Spalte {col} für Treatment nicht in X gefunden.")
    
        if base_values is None:
            base_values = [0] * len(treat_cols) # Standard-Basiswerte sind 0
    
        for col, val in zip(treat_cols, base_values):
            if col in X_base.columns:
                X_base[col] = val
            else:
                self.logger.warning(f"Spalte {col} für Basis nicht in X gefunden.")
        
        # Die predict-Methode kümmert sich um das Hinzufügen aller Interaktionsterme
        y_treat = self.predict(X_treat)
        y_base = self.predict(X_base)
        
        return (y_treat - y_base).values