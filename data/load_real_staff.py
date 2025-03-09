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
SENSITIVE_COL = "female"
COVARIATE_COLS = ["age", "educ", "white", "black", "hispanic", "english", "cohabmarried", "haschild", "everwkd", "mwearn", "hhsize", "educmum", "educdad", "welfarechild", "health", "smoke", "alcohol"]
CATEGORICAL_COLS = ["white", "black", "hispanic", "english", "cohabmarried", "haschild", "everwkd"]
CONTINUOUS_COLS = ["age", "educ", "mwearn", "hhsize", "educmum", "educdad"]
ORDINAL_COLS = ["welfarechild", "health", "smoke", "alcohol"]

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from an RData file."""
    try:
        data = pyreadr.read_r(file_path)[None]
    except Exception as e:
        print(f"Error: Failed to load RData file at {file_path}. {str(e)}")
        return None
    return data

def preprocess_data(data: pd.DataFrame) -> tuple:
    """Preprocess data by removing missing values and encoding categorical variables."""
    data = data.dropna(subset=[OUTCOME_COL, TREAT_COL] + COVARIATE_COLS)
    # Train/Val/Test split
    f_train = 0.7
    f_val = 0.15
    df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42), [int(f_train * len(data)), int((f_train + f_val) * len(data))])

    # Outcome + treatment
    y_train = np.expand_dims(df_train[OUTCOME_COL].values, axis=1)
    a_train = np.expand_dims(df_train[TREAT_COL].values, axis=1)
    y_val = np.expand_dims(df_val[OUTCOME_COL].values, axis=1)
    a_val = np.expand_dims(df_val[TREAT_COL].values, axis=1)
    y_test = np.expand_dims(df_test[OUTCOME_COL].values, axis=1)
    a_test = np.expand_dims(df_test[TREAT_COL].values, axis=1)

    # Sensitives Attribut
    s_train = np.expand_dims(df_train[SENSITIVE_COL].values, axis=1)
    s_val = np.expand_dims(df_val[SENSITIVE_COL].values, axis=1)
    s_test = np.expand_dims(df_test[SENSITIVE_COL].values, axis=1)
    enc_s = OneHotEncoder().fit(s_train)
    s_train = enc_s.transform(s_train).toarray()
    s_val = enc_s.transform(s_val).toarray()
    s_test = enc_s.transform(s_test).toarray()

    # Kovariaten
    x_train = df_train[COVARIATE_COLS].values
    x_val = df_val[COVARIATE_COLS].values
    x_test = df_test[COVARIATE_COLS].values

    return y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test

def create_datasets(y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test) -> dict:
    """Create datasets from preprocessed data."""
    # Kovariatentypen
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

    # Datasets erstellen
    d_train = Static_Dataset(y=y_train, a=a_train, x=x_train, s=s_train, y_type="continuous", x_type=x_types, s_type=["categorical"])
    d_val = Static_Dataset(y=y_val, a=a_val, x=x_val, s=s_val, y_type="continuous", x_type=x_types, s_type=["categorical"])
    d_test = Static_Dataset(y=y_test, a=a_test, x=x_test, s=s_test, y_type="continuous", x_type=x_types, s_type=["categorical"])

    # Standardisierung und Tensor-Konvertierung
    d_train.standardize()
    d_val.standardize()
    d_test.standardize()
    d_train.convert_to_tensor()
    d_val.convert_to_tensor()
    d_test.convert_to_tensor()

    return {"d_train": d_train, "d_val": d_val, "d_test": d_test}

def main():
    # Konfiguration anpassen
    config = {"train_frac": 0.7, "val_frac": 0.15}
    project_path = utils.get_project_path()
    path = os.path.join(project_path, "data", "JC.RData")
    data = load_data(path)

    if data: # überprüft ob die Daten erfolgreich geladen wurden.
        print("Job Corps-Daten erfolgreich geladen und verarbeitet.")
        y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test = preprocess_data(data)
        datasets = create_datasets(y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test)
        print(datasets["d_train"].data["y"].shape) # zeigt die Dimensionen der geladenen Daten.
    else:
        print("Fehler beim Laden der Daten!")

if __name__ == "__main__":
    main()
