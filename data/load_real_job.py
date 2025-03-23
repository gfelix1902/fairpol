import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import IterativeImputer
from data.data_structures import Static_Dataset

def load_data_from_csv(config=None, standardize=True):
    print("🔎 Lade Daten aus CSV...")

    # Daten laden
    try:
        data = pd.read_csv("JC_processed.csv")
        print(f"✅ Daten erfolgreich geladen. Shape: {data.shape}")
    except FileNotFoundError:
        raise FileNotFoundError("❌ Datei 'JC_processed.csv' nicht gefunden!")
    
    # Sicherstellen, dass die 'outcome' Spalte existiert
    if 'earny4' not in data.columns:
        raise ValueError("❌ Spalte 'earny4' fehlt im Datensatz. Stelle sicher, dass sie vorhanden ist.")
    
    # Vor der Imputation
    print("🔍 Vor der Imputation:")
    print(data.isna().sum())  # Anzahl der NaN-Werte pro Spalte

    # Imputation durchführen
    imp = IterativeImputer(max_iter=10, random_state=0)
    data_imputed = pd.DataFrame(imp.fit_transform(data), columns=data.columns)

    # Nach der Imputation
    print("🔍 Nach der Imputation:")
    print(data_imputed.isna().sum())  # Noch NaN-Werte?

    # Prüfen, ob imputed DataFrame irgendwelche NaN-Werte enthält
    if data_imputed.isna().any().any():
        print("❌ Es sind immer noch NaN-Werte vorhanden!")
    else:
        print("✅ Keine NaN-Werte mehr nach der Imputation.")

    # Outcome + Treatment extrahieren
    y = np.expand_dims(data["earny4"].values, axis=1) 
    a = np.expand_dims(data["assignment"].values, axis=1)

    # Sicherstellen, dass die Dimensionen stimmen
    if y.shape[0] == 0 or a.shape[0] == 0:
        raise ValueError("❌ Outcome- oder Treatment-Daten sind leer!")
    
    # Sensitive Attribute (z.B. Geschlecht)
    s = np.expand_dims(data["female"].values, axis=1)
    enc_s = OneHotEncoder(sparse_output=False)  # Verwende sparse_output=False
    s = enc_s.fit_transform(s)

    # Kategorische Spalten für One-Hot-Encoding
    categorical_columns = ['white', 'black', 'hispanic', 'english', 'cohabmarried', 'haschild']

    # One-Hot-Encoding für die kategorischen Spalten durchführen
    encoder = OneHotEncoder(sparse_output=False)  # Verwende sparse_output=False
    encoded_data = encoder.fit_transform(data[categorical_columns])

    # Namen der neuen Spalten für das One-Hot-Encoding
    encoded_columns = encoder.get_feature_names_out(categorical_columns)

    # One-Hot-encodierte Daten als DataFrame erstellen
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

    # Entferne die ursprünglichen kategorischen Spalten und füge die encodierten Spalten hinzu
    data_encoded = pd.concat([data.drop(columns=categorical_columns), encoded_df], axis=1)

    # Ausgabe der Daten nach One-Hot-Encoding
    print("🔑 Daten nach One-Hot-Encoding:")
    print(data_encoded.head())

    # Unabhängige Variablen (Features) anpassen
    # Füge die neuen One-Hot-kodierten Spalten zu x_names hinzu
    x_names = ["age", "educ", "geddegree", "hsdegree", "mwearn", "hhsize",
            "white_0.0", "white_1.0", "black_0.0", "black_1.0", "hispanic_0.0", "hispanic_1.0",
            "english_0.0", "english_1.0", "cohabmarried_0.0", "cohabmarried_1.0",
            "haschild_0.0", "haschild_1.0"]
    x_types = ["continuous", "continuous", "binary", "binary", "continuous", "continuous"] + ["categorical"] * 12
    
    # Die Features aus den vollständigen, kodierten Daten extrahieren
    x = data_encoded[x_names].values
    if x.shape[0] == 0:
        raise ValueError("❌ Feature-Daten sind leer!")
    
    # Outcome-Typ festlegen
    outcome_type = "continuous"

    # Train/Val/Test Split
    f_train = 0.7
    f_val = 0.15
    df_train, df_val, df_test = np.split(data_encoded.sample(frac=1, random_state=42), 
                                         [int(f_train * len(data_encoded)), int((f_train + f_val) * len(data_encoded))])

    print(f"✅ Train/Val/Test-Split: {len(df_train)}/{len(df_val)}/{len(df_test)}")

    # Trainingsdaten
    y_train = np.expand_dims(df_train["earny4"].values, axis=1)
    a_train = np.expand_dims(df_train["assignment"].values, axis=1)
    x_train = df_train[x_names].values
    s_train = enc_s.transform(np.expand_dims(df_train["female"].values, axis=1))

    # Validierungsdaten
    y_val = np.expand_dims(df_val["earny4"].values, axis=1)
    a_val = np.expand_dims(df_val["assignment"].values, axis=1)
    x_val = df_val[x_names].values
    s_val = enc_s.transform(np.expand_dims(df_val["female"].values, axis=1))

    # Testdaten
    y_test = np.expand_dims(df_test["earny4"].values, axis=1)
    a_test = np.expand_dims(df_test["assignment"].values, axis=1)
    x_test = df_test[x_names].values
    s_test = enc_s.transform(np.expand_dims(df_test["female"].values, axis=1))

    # Debug-Ausgaben für Dimensionen
    print(f"📏 Dimensionen der Trainingsdaten: y={y_train.shape}, a={a_train.shape}, x={x_train.shape}, s={s_train.shape}")
    print(f"📏 Dimensionen der Testdaten: y={y_test.shape}, a={a_test.shape}, x={x_test.shape}, s={s_test.shape}")

    # Static_Dataset erzeugen
    d_train = Static_Dataset(y=y_train, a=a_train, x=x_train, s=s_train, 
                             y_type=outcome_type, x_type=x_types, s_type=["categorical"])
    d_val = Static_Dataset(y=y_val, a=a_val, x=x_val, s=s_val, 
                           y_type=outcome_type, x_type=x_types, s_type=["categorical"])
    d_test = Static_Dataset(y=y_test, a=a_test, x=x_test, s=s_test, 
                            y_type=outcome_type, x_type=x_types, s_type=["categorical"])

    # Nullwerte prüfen (Problemquelle für df_results!)
    if any(d is None for d in [d_train, d_val, d_test]):
        raise ValueError("❌ Einer der Datensätze (Train/Val/Test) ist leer!")

    if standardize:
        print("📏 Standardisiere die Daten...")
        d_train.standardize()
        d_val.standardize()
        d_test.standardize()

    # In Tensor umwandeln (wichtig für PyTorch)
    print("🔄 Konvertiere Daten zu Torch-Tensoren...")
    d_train.convert_to_tensor()
    d_val.convert_to_tensor()
    d_test.convert_to_tensor()

    print("✅ Daten wurden erfolgreich geladen und verarbeitet!")

    return {"d_train": d_train, "d_val": d_val, "d_test": d_test}

# Debugging-Einstiegspunkt
if __name__ == "__main__":
    try:
        datasets = load_data_from_csv()
        print("🚀 Datensätze erfolgreich geladen!")
    except Exception as e:
        print(f"❌ Fehler beim Laden der Daten: {e}")
