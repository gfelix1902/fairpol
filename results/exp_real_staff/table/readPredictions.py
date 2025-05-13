import utils
import pandas as pd
import numpy as np
import joblib

def print_df(title, df):
    print(f"\n{'-'*40}\n{title}\n{'-'*40}")
    print(df.round(4))

if __name__ == "__main__":
    path = utils.get_project_path() + "/results/exp_real_staff/table/"

    # Load predictions.pkl
    try:
        predictions = joblib.load(path + "predictions.pkl")
        print("✅ Daten erfolgreich aus der predictions.pkl-Datei geladen!")
    except FileNotFoundError:
        print(f"❌ Fehler: Datei nicht gefunden unter {path + 'predictions.pkl'}.")
        exit()
    except Exception as e:
        print(f"❌ Fehler beim Laden der predictions.pkl-Datei: {e}")
        exit()

    # Predictions: Mean and SD
    predictions["index"] = predictions.index
    pred_means = predictions.groupby('index').mean().drop(columns=["run"])
    pred_sds = predictions.groupby('index').std().drop(columns=["run"])

    print(predictions.groupby('index').size().value_counts())

    print(predictions[predictions['index'] == 0]['ols'])

    # Formattierte Ausgabe
    print_df("Predictions Mean", pred_means)
    print_df("Predictions SD", pred_sds)

    # Column-Wise Aggregated Mean and SD
    column_aggregated_means = pred_means.mean()
    column_aggregated_sds = pred_sds.mean()

    print("\n📊 Column-Wise Aggregated Values")
    for column in column_aggregated_means.index:
        print(f"{column}: Mean = {column_aggregated_means[column]:.4f}, SD = {column_aggregated_sds[column]:.4f}")