import numpy as np
import pandas as pd
import utils
from data.data_structures import Static_Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

def load_job_corps(config, standardize=True):
    # Datenpfad anpassen
    project_path = utils.get_project_path()
    path = os.path.join(project_path, "data", "JC.csv")  # Ersetzen Sie "JC.csv" durch Ihren Dateinamen
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden unter {path}. Stellen Sie sicher, dass die Datei vorhanden ist.")
        return None

    # Spaltennamen und Datentypen anpassen
    outcome_col = "earny4"  # Beispiel: Weekly earnings in fourth year
    treat_col = "assignment"
    sensitive_col = "female"
    covariate_cols = ["age", "educ", "white", "black", "hispanic", "english", "cohabmarried", "haschild", "everwkd", "mwearn", "hhsize", "educmum", "educdad", "welfarechild", "health", "smoke", "alcohol"]
    categorical_cols = ["white", "black", "hispanic", "english", "cohabmarried", "haschild", "everwkd"]
    continuous_cols = ["age", "educ", "mwearn", "hhsize", "educmum", "educdad"]
    ordinal_cols = ["welfarechild", "health", "smoke", "alcohol"]

    outcome_type = "continuous"

    # Datenvorverarbeitung
    data = data.dropna(subset=[outcome_col, treat_col] + covariate_cols) # Entfernt Zeilen mit NAs in relevanten Spalten
    # Train/ Val/ Test split
    f_train = config["train_frac"]
    f_val = config["val_frac"]
    df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42), [int(f_train * len(data)), int((f_train + f_val) * len(data))])

    # Outcome + treatment
    y_train = np.expand_dims(df_train[outcome_col].values, axis=1)
    a_train = np.expand_dims(df_train[treat_col].values, axis=1)
    y_val = np.expand_dims(df_val[outcome_col].values, axis=1)
    a_val = np.expand_dims(df_val[treat_col].values, axis=1)
    y_test = np.expand_dims(df_test[outcome_col].values, axis=1)
    a_test = np.expand_dims(df_test[treat_col].values, axis=1)

    # Sensitives Attribut
    s_train = np.expand_dims(df_train[sensitive_col].values, axis=1)
    s_val = np.expand_dims(df_val[sensitive_col].values, axis=1)
    s_test = np.expand_dims(df_test[sensitive_col].values, axis=1)
    enc_s = OneHotEncoder().fit(s_train)
    s_train = enc_s.transform(s_train).toarray()
    s_val = enc_s.transform(s_val).toarray()
    s_test = enc_s.transform(s_test).toarray()

    # Kovariaten
    x_train = df_train[covariate_cols].values
    x_val = df_val[covariate_cols].values
    x_test = df_test[covariate_cols].values

    # Kovariatentypen
    x_types = []
    for col in covariate_cols:
        if col in continuous_cols:
            x_types.append("continuous")
        elif col in categorical_cols:
            x_types.append("categorical")
        elif col in ordinal_cols:
            x_types.append("ordinal")
        else:
            x_types.append("unknown")

    # Datasets erstellen
    d_train = Static_Dataset(y=y_train, a=a_train, x=x_train, s=s_train, y_type=outcome_type, x_type=x_types, s_type=["categorical"])
    d_val = Static_Dataset(y=y_val, a=a_val, x=x_val, s=s_val, y_type=outcome_type, x_type=x_types, s_type=["categorical"])
    d_test = Static_Dataset(y=y_test, a=a_test, x=x_test, s=s_test, y_type=outcome_type, x_type=x_types, s_type=["categorical"])

    # Standardisierung und Tensor-Konvertierung
    if standardize:
        d_train.standardize()
        d_val.standardize()
        d_test.standardize()
    d_train.convert_to_tensor()
    d_val.convert_to_tensor()
    d_test.convert_to_tensor()

    return {"d_train": d_train, "d_val": d_val, "d_test": d_test}

if __name__ == "__main__":
    # Konfiguration anpassen
    config = {"train_frac": 0.7, "val_frac": 0.15}
    data = load_job_corps(config)

    if data: # überprüft ob die Daten erfolgreich geladen wurden.
        print("Job Corps-Daten erfolgreich geladen und verarbeitet.")
        # Hier können Sie den Code hinzufügen, der die Datensätze verwendet (z. B. Modelltraining).
        print(data["d_train"].data["y"].shape) # zeigt die Dimensionen der geladenen Daten.