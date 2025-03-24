import numpy as np
import pandas as pd
import utils
from data.data_structures import Static_Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
import pyreadr

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

def load_data() -> pd.DataFrame:
    """Load data from an RData file."""
    project_path = utils.get_project_path()
    path = os.path.join(project_path, "JC_processed.csv")  
    
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden unter {path}. Stellen Sie sicher, dass die Datei vorhanden ist.")
        return None

def preprocess_data(data: pd.DataFrame) -> tuple:
    """Preprocess data by removing sensitive attributes and encoding categorical variables."""
    
    # Fehlende Werte entfernen
    data = data.dropna(subset=[OUTCOME_COL, TREAT_COL] + COVARIATE_COLS)

    # Train/Val/Test Split
    f_train = 0.7
    f_val = 0.15
    df_train, df_val, df_test = np.split(
        data.sample(frac=1, random_state=42), 
        [int(f_train * len(data)), int((f_train + f_val) * len(data))]
    )

    # Outcome + Treatment
    y_train = np.expand_dims(df_train[OUTCOME_COL].values, axis=1)
    a_train = np.expand_dims(df_train[TREAT_COL].values, axis=1)
    y_val = np.expand_dims(df_val[OUTCOME_COL].values, axis=1)
    a_val = np.expand_dims(df_val[TREAT_COL].values, axis=1)
    y_test = np.expand_dims(df_test[OUTCOME_COL].values, axis=1)
    a_test = np.expand_dims(df_test[TREAT_COL].values, axis=1)

    # ➡️ Sensible Attribute entfernen
    df_train = df_train.drop(columns=[col for col in SENSITIVE_ATTRS if col in df_train.columns])
    df_val = df_val.drop(columns=[col for col in SENSITIVE_ATTRS if col in df_val.columns])
    df_test = df_test.drop(columns=[col for col in SENSITIVE_ATTRS if col in df_test.columns])

    # ➡️ Kovariaten definieren (ohne sensible Attribute)
    x_train = df_train[COVARIATE_COLS].values
    x_val = df_val[COVARIATE_COLS].values
    x_test = df_test[COVARIATE_COLS].values

    # ➡️ Sensible Attribute separat speichern (für Fairness-Analyse)
    s_train = df_train[SENSITIVE_ATTRS].values if all(col in df_train.columns for col in SENSITIVE_ATTRS) else None
    s_val = df_val[SENSITIVE_ATTRS].values if all(col in df_val.columns for col in SENSITIVE_ATTRS) else None
    s_test = df_test[SENSITIVE_ATTRS].values if all(col in df_test.columns for col in SENSITIVE_ATTRS) else None

    return y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test

def create_datasets(y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test) -> dict:
    """Create datasets from preprocessed data."""
    
    # Kovariatentypen definieren
    x_types = []
    for col in COVARIATE_COLS:
        if col in CONTINUOUS_COLS:
            x_types.append("continuous")
        elif col in CATEGORICAL_COLS:
            x_types.append("categorical")
        elif col in ORDINAL_COLS:
            x_types.append("ordinal")
        else:
            x_types.append("unknown")

    # ➡️ Datasets ohne sensitive Attribute erstellen
    d_train = Static_Dataset(y=y_train, a=a_train, x=x_train, y_type="continuous", x_type=x_types)
    d_val = Static_Dataset(y=y_val, a=a_val, x=x_val, y_type="continuous", x_type=x_types)
    d_test = Static_Dataset(y=y_test, a=a_test, x=x_test, y_type="continuous", x_type=x_types)

    # ➡️ Standardisierung und Tensor-Konvertierung
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

def main(config_data):
    """Main function to load, preprocess, and create datasets."""
    data = load_data()

    if data is not None:
        print("Job Corps-Daten erfolgreich geladen und verarbeitet.")
        y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test = preprocess_data(data)
        datasets = create_datasets(y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test)
        
        print(datasets["d_train"].data["y"].shape)  # zeigt die Dimensionen der geladenen Daten.
        return datasets
    else:
        print("Fehler beim Laden der Daten!")
        return None

if __name__ == "__main__":
    main({})
