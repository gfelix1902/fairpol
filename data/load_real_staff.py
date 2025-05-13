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

def preprocess_data(data: pd.DataFrame, seed=None) -> tuple:
    """Preprocess data by encoding and handling missing values."""

    # Fehlende Werte mit IterativeImputer auffüllen
    imp = IterativeImputer(max_iter=10, random_state=0)
    data_imputed = pd.DataFrame(imp.fit_transform(data), columns=data.columns)

    print("\n✅ Keine NaN-Werte mehr nach der Imputation.")

    # 🔎 2. Verteilung der wichtigsten Features überprüfen
    plt.figure(figsize=(14, 10))
    for i, col in enumerate(CONTINUOUS_COLS[:6]):
        plt.subplot(3, 2, i + 1)
        sns.histplot(data_imputed[col], kde=True)
        plt.title(f"Verteilung von {col}")
    plt.tight_layout()
    plt.show()

    # 🔎 3. Kategoriale Werte überprüfen
    for col in CATEGORICAL_COLS:
        print(f"\n🔑 Werte in '{col}':")
        print(data_imputed[col].value_counts())

    # Train/Val/Test Split
    df_train, df_val, df_test = split_data(data_imputed, seed=seed)

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


def split_data(data: pd.DataFrame, seed=None) -> tuple:
    f_train = 0.7
    f_val = 0.15
    df_train, df_val, df_test = np.split(
        data.sample(frac=1, random_state=seed),
        [int(f_train * len(data)), int((f_train + f_val) * len(data))]
    )
    return df_train, df_val, df_test

def main(config_data, seed=None):
    """Main function to load and preprocess the data."""
    data = load_data()

    if data is not None:
        print("\n🚀 Beginne Preprocessing...")
        y_train, a_train, s_train, x_train, y_val, a_val, s_val, x_val, y_test, a_test, s_test, x_test = preprocess_data(data, seed=seed)
        print("\n✅ Daten wurden erfolgreich geladen und verarbeitet!")
    else:
        print("\n❌ Fehler beim Laden der Daten!")

if __name__ == "__main__":
    main()
