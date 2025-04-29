import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
import utils
import seaborn as sns
import matplotlib.pyplot as plt

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
    """Load data from a CSV file."""
    project_path = utils.get_project_path()
    path = os.path.join(project_path, "JC_processed.csv")

    try:
        print(f"🔎 Lade Daten aus CSV von: {path}")
        data = pd.read_csv(path)
        print(f"✅ Daten erfolgreich geladen. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"❌ Fehler: Datei nicht gefunden unter {path}.")
        return None

def preprocess_data(data: pd.DataFrame) -> tuple:
    """Preprocess data by encoding and handling missing values."""

    # Fehlende Werte mit IterativeImputer auffüllen
    imp = IterativeImputer(max_iter=10, random_state=0)
    data = pd.DataFrame(imp.fit_transform(data), columns=data.columns)

    print("\n✅ Keine NaN-Werte mehr nach der Imputation.")

    # 🔎 2. Verteilung der wichtigsten Features überprüfen
    plt.figure(figsize=(14, 10))
    for i, col in enumerate(CONTINUOUS_COLS[:6]):
        plt.subplot(3, 2, i + 1)
        sns.histplot(data[col], kde=True)
        plt.title(f"Verteilung von {col}")
    plt.tight_layout()
    plt.show()

    # 🔎 3. Kategoriale Werte überprüfen
    for col in CATEGORICAL_COLS:
        print(f"\n🔑 Werte in '{col}':")
        print(data[col].value_counts())

    # Train/Val/Test Split
    f_train = 0.7
    f_val = 0.15
    df_train, df_val, df_test = np.split(
        data.sample(frac=1, random_state=42),
        [int(f_train * len(data)), int((f_train + f_val) * len(data))]
    )

    print(f"\n✅ Train/Val/Test-Split: {len(df_train)}/{len(df_val)}/{len(df_test)}")

    # Outcome + Treatment
    y_train = np.expand_dims(df_train[OUTCOME_COL].values, axis=1)
    a_train = np.expand_dims(df_train[TREAT_COL].values, axis=1)
    y_val = np.expand_dims(df_val[OUTCOME_COL].values, axis=1)
    a_val = np.expand_dims(df_val[TREAT_COL].values, axis=1)
    y_test = np.expand_dims(df_test[OUTCOME_COL].values, axis=1)
    a_test = np.expand_dims(df_test[TREAT_COL].values, axis=1)

    # ➡️ Sensible Attribute separat speichern (für Fairness-Analyse)
    s_train = df_train[SENSITIVE_ATTRS].values
    s_val = df_val[SENSITIVE_ATTRS].values
    s_test = df_test[SENSITIVE_ATTRS].values

    # ➡️ Kovariaten definieren
    x_train = df_train[COVARIATE_COLS].values
    x_val = df_val[COVARIATE_COLS].values
    x_test = df_test[COVARIATE_COLS].values

    print(f"\n📏 Dimensionen der Trainingsdaten: y={y_train.shape}, a={a_train.shape}, x={x_train.shape}, s={s_train.shape}")
    print(f"📏 Dimensionen der Testdaten: y={y_test.shape}, a={a_test.shape}, x={x_test.shape}, s={s_test.shape}")

    return y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test

def calculate_statistics(sensitive_attrs, predictions, attribute_names):
    """
    Berechnet Mean, STD und andere Statistiken für jedes sensitive Attribute.

    Args:
        sensitive_attrs (np.ndarray): Array mit sensitiven Attributen (z. B. female, white, black, hispanic).
        predictions (np.ndarray): Array mit Werten (z. B. tau_hat oder andere Vorhersagen).
        attribute_names (list): Liste der sensitiven Attributnamen.

    Returns:
        dict: Statistiken für jedes sensitive Attribut.
    """
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

def main():
    """Main function to load and preprocess the data."""
    data = load_data()

    if data is not None:
        print("\n🚀 Beginne Preprocessing...")
        y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test = preprocess_data(data)
        print("\n✅ Daten wurden erfolgreich geladen und verarbeitet!")

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
        print("\n❌ Fehler beim Laden der Daten!")

if __name__ == "__main__":
    main()
