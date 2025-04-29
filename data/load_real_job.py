import numpy as np
import pandas as pd
import utils
from data.data_structures import Static_Dataset
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

# Define constant values
OUTCOME_COL = "earny4"
TREAT_COL = "assignment"

# Sensible Attribute
SENSITIVE_ATTRS = ["female", "white", "black", "hispanic"]

# Kovariaten ohne sensible Attribute
COVARIATE_COLS = ["age", "educ", "english", "cohabmarried", "haschild",
                  "everwkd", "mwearn", "hhsize", "educmum", "educdad",
                  "welfarechild", "health", "smoke", "alcohol"]

CATEGORICAL_COLS = ["english", "cohabmarried", "haschild", "everwkd"]
CONTINUOUS_COLS = ["age", "educ", "mwearn", "hhsize", "educmum", "educdad"]
ORDINAL_COLS = ["welfarechild", "health", "smoke", "alcohol"]

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden unter {filepath}. Stellen Sie sicher, dass die Datei vorhanden ist.")
        return None

def impute_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using IterativeImputer."""
    imp = IterativeImputer(max_iter=10, random_state=0)
    data_imputed = pd.DataFrame(imp.fit_transform(data), columns=data.columns)
    return data_imputed

def validate_imputed_data(data: pd.DataFrame) -> None:
    """Validate imputed data for NaN and infinite values."""
    if data.isna().any().any():
        print("❌ Es sind immer noch NaN-Werte vorhanden!")
    if np.isinf(data.values).any():
        print("❌ Unendliche Werte nach der Imputation vorhanden!")
    if (data.values < -1e10).any() or (data.values > 1e10).any():
        print("❌ Sehr große/kleine Werte nach der Imputation gefunden!")

def split_data(data: pd.DataFrame) -> tuple:
    """Split data into train, validation, and test sets."""
    f_train = 0.7
    f_val = 0.15
    df_train, df_val, df_test = np.split(
        data.sample(frac=1, random_state=42),
        [int(f_train * len(data)), int((f_train + f_val) * len(data))]
    )
    return df_train, df_val, df_test

def extract_features_and_targets(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    """Extract features and targets from train, validation, and test sets."""
    y_train = df_train[OUTCOME_COL].values.reshape(-1, 1)
    a_train = df_train[TREAT_COL].values.reshape(-1, 1)
    y_val = df_val[OUTCOME_COL].values.reshape(-1, 1)
    a_val = df_val[TREAT_COL].values.reshape(-1, 1)
    y_test = df_test[OUTCOME_COL].values.reshape(-1, 1)
    a_test = df_test[TREAT_COL].values.reshape(-1, 1)

    x_train = df_train[COVARIATE_COLS].values
    x_val = df_val[COVARIATE_COLS].values
    x_test = df_test[COVARIATE_COLS].values

    s_train = df_train[SENSITIVE_ATTRS].values if all(col in df_train.columns for col in SENSITIVE_ATTRS) else np.empty((y_train.shape[0], 0))
    s_val = df_val[SENSITIVE_ATTRS].values if all(col in df_val.columns for col in SENSITIVE_ATTRS) else np.empty((y_val.shape[0], 0))
    s_test = df_test[SENSITIVE_ATTRS].values if all(col in df_test.columns for col in SENSITIVE_ATTRS) else np.empty((y_test.shape[0], 0))

    return y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test

def preprocess_data(data: pd.DataFrame) -> tuple:
    """Preprocess data: remove missing values, impute, and split into train/val/test."""

    #print(f"Anzahl der Zeilen vor dem Entfernen von fehlenden Werten: {len(data)}")

    data = data.dropna(subset=[OUTCOME_COL, TREAT_COL] + COVARIATE_COLS)

    #print(f"Anzahl der Zeilen nach dem Entfernen von fehlenden Werten: {len(data)}")

    if data.empty:
        print("Warnung: Daten sind nach dem Entfernen von fehlenden Werten leer!")
        return None, None, None, None, None, None, None, None, None, None, None, None

    data_imputed = impute_missing_values(data)
    validate_imputed_data(data_imputed)

    df_train, df_val, df_test = split_data(data_imputed)
    return extract_features_and_targets(df_train, df_val, df_test)

def create_datasets(y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test) -> dict:
    x_types = [
        "continuous" if col in CONTINUOUS_COLS else
        "categorical" if col in CATEGORICAL_COLS else
        "ordinal" if col in ORDINAL_COLS else
        "unknown"
        for col in COVARIATE_COLS
    ]

    s_type = ["categorical"] * s_train.shape[1]

    d_train = Static_Dataset(y_train, a_train, x_train, s_train, "continuous", x_types, s_type)
    d_val = Static_Dataset(y_val, a_val, x_val, s_val, "continuous", x_types, s_type)
    d_test = Static_Dataset(y_test, a_test, x_test, s_test, "continuous", x_types, s_type)

    d_train.standardize()
    d_val.standardize()
    d_test.standardize()

    d_train.convert_to_tensor()
    d_val.convert_to_tensor()
    d_test.convert_to_tensor()

    return {
        "d_train": d_train,
        "d_val": d_val,
        "d_test": d_test
    }

def calculate_statistics(sensitive_attrs, predictions, attribute_names):
    statistics = {}

    for i, attr in enumerate(attribute_names):
        print(f"\n🔍 Berechnung der Statistiken für: {attr}")

        # Werte für die Gruppen (z. B. 0 und 1) extrahieren
        group_0 = predictions[sensitive_attrs[:, i] == 0]
        group_1 = predictions[sensitive_attrs[:, i] == 1]

        # Berechnung von Mean und STD
        mean_0, std_0 = group_0.mean(), group_0.std()
        mean_1, std_1 = group_1.mean(), group_1.std()

        # Berechnung der Differenz
        mean_diff = abs(mean_1 - mean_0)

        # Speichern der Ergebnisse
        statistics[attr] = {
            "mean_0": mean_0,
            "std_0": std_0,
            "mean_1": mean_1,
            "std_1": std_1,
            "mean_diff": mean_diff
        }

        # Ausgabe der Ergebnisse
        print(f"  Gruppe 0 (mean={mean_0:.4f}, std={std_0:.4f})")
        print(f"  Gruppe 1 (mean={mean_1:.4f}, std={std_1:.4f})")
        print(f"  Differenz der Mittelwerte: {mean_diff:.4f}")

    return statistics

def main(config_data):
    try:
        project_path = utils.get_project_path()
        path = os.path.join(project_path, "JC_processed.csv")
        data = load_data(path)
        if data is not None:
            y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test = preprocess_data(data)
            datasets = create_datasets(y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test)

            # Beispiel: Vorhersagen (tau_hat) generieren
            predictions = np.random.rand(len(s_test))  # Ersetze dies durch deine tatsächlichen Vorhersagen

            # Statistiken für sensitive Attribute berechnen
            statistics = calculate_statistics(s_test, predictions, SENSITIVE_ATTRS)

            # Ergebnisse speichern oder weiterverarbeiten
            import json
            with open("sensitive_statistics.json", "w") as f:
                json.dump(statistics, f, indent=4)

            print("\n✅ Statistiken für sensitive Attribute berechnet und gespeichert in 'sensitive_statistics.json'.")
        else:
            print("Fehler beim Laden der Daten!")
    except Exception as e:
        print(f"Fehler während der Datenverarbeitung: {e}")