import utils
import pandas as pd
import numpy as np
import joblib

if __name__ == "__main__":
    path = utils.get_project_path() + "/results/exp_real_staff/table/"

    # Load data
    config_exp = utils.load_yaml("/experiments/exp_real_staff/config_real_staff")
    config_data = config_exp["data"]
    datasets = utils.load_data(config_data)
    predictions = []
    pvalues = []
    pvalues1 = []
    pvalues0 = []
    af = []
    predictions = joblib.load(path + "predictions.pkl")
    pvalues = joblib.load(path + "pvalues.pkl")
    pvalues1 = joblib.load(path + "pvalues1.pkl")
    pvalues0 = joblib.load(path + "pvalues0.pkl")
    af = joblib.load(path + "af.pkl")

    # Funktion zum Entpacken von NumPy-Arrays in DataFrames
    def unpack_numpy_arrays(df):
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, np.ndarray) and x.ndim > 0 else x)
        return df

    # Entpacke die NumPy-Arrays
    pvalues = unpack_numpy_arrays(pvalues)
    pvalues1 = unpack_numpy_arrays(pvalues1)
    pvalues0 = unpack_numpy_arrays(pvalues0)
    af = unpack_numpy_arrays(af)

    # Policy values
    pvalues_means = pvalues.mean().to_frame().transpose()
    pvalues_sds = pvalues.std().to_frame().transpose()
    pvalues1_means = pvalues1.mean().to_frame().transpose()
    pvalues1_sds = pvalues1.std().to_frame().transpose()
    pvalues0_means = pvalues0.mean().to_frame().transpose()
    pvalues0_sds = pvalues0.std().to_frame().transpose()
    af_means = af.mean().to_frame().transpose()
    af_sds = af.std().to_frame().transpose()

    # Predictions
    predictions["index"] = predictions.index
    pred_means = predictions.groupby('index').mean().drop(columns=["run"])
    pred_sds = predictions.groupby('index').std().drop(columns=["run"])

    # Printing
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Pvalues Mean")
        print(pvalues_means)
        print("Pvalues SD")
        print(pvalues_sds)
        print("Pvalues 1 Mean")
        print(pvalues1_means)
        print("Pvalues 1 SD")
        print(pvalues1_sds)
        print("Pvalues 0 Mean")
        print(pvalues0_means)
        print("Pvalues 0 SD")
        print(pvalues0_sds)
        print("AF Mean")
        print(af_means)
        print("AF SD")
        print(af_sds)