import utils
import pandas as pd
import numpy as np
import joblib
import os

def load_and_unpack_data(path):
    """Loads and unpacks data from pickle files."""
    try:
        predictions = joblib.load(os.path.join(path, "predictions.pkl"))
        pvalues = joblib.load(os.path.join(path, "pvalues.pkl"))
        pvalues1 = joblib.load(os.path.join(path, "pvalues1.pkl"))
        pvalues0 = joblib.load(os.path.join(path, "pvalues0.pkl"))
        af = joblib.load(os.path.join(path, "af.pkl"))
        return predictions, pvalues, pvalues1, pvalues0, af
    except FileNotFoundError as e:
        print(f"Error loading pickle files: {e}")
        return None, None, None, None, None

def unpack_numpy_arrays(df):
    """Unpacks numpy arrays within a DataFrame."""
    if isinstance(df, pd.DataFrame):
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, np.ndarray) and x.ndim > 0 else x)
        return df
    else:
        print("Error: Input is not a Pandas DataFrame.")
        return None

def calculate_and_print_stats(predictions, pvalues, pvalues1, pvalues0, af):
    """Calculates and prints statistics."""
    if pvalues is not None:
        pvalues_means = pvalues.mean().to_frame().transpose()
        pvalues_sds = pvalues.std().to_frame().transpose() if len(pvalues) > 1 else pd.DataFrame(np.nan, index=[0], columns=pvalues.columns)
        pvalues1_means = pvalues1.mean().to_frame().transpose()
        pvalues1_sds = pvalues1.std().to_frame().transpose() if len(pvalues1) > 1 else pd.DataFrame(np.nan, index=[0], columns=pvalues1.columns)
        pvalues0_means = pvalues0.mean().to_frame().transpose()
        pvalues0_sds = pvalues0.std().to_frame().transpose() if len(pvalues0) > 1 else pd.DataFrame(np.nan, index=[0], columns=pvalues0.columns)
        af_means = af.mean().to_frame().transpose()
        af_sds = af.std().to_frame().transpose() if len(af) > 1 else pd.DataFrame(np.nan, index=[0], columns=af.columns)

        predictions["index"] = predictions.index
        pred_means = predictions.groupby('index').mean().drop(columns=["run"])
        pred_sds = predictions.groupby('index').std().drop(columns=["run"])

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print("Pvalues Mean\n", pvalues_means)
            print("Pvalues SD\n", pvalues_sds)
            print("Pvalues 1 Mean\n", pvalues1_means)
            print("Pvalues 1 SD\n", pvalues1_sds)
            print("Pvalues 0 Mean\n", pvalues0_means)
            print("Pvalues 0 SD\n", pvalues0_sds)
            print("AF Mean\n", af_means)
            print("AF SD\n", af_sds)
            
if __name__ == "__main__":
    path = utils.get_project_path() + "/results/exp_real_staff/table/"
    predictions, pvalues, pvalues1, pvalues0, af = load_and_unpack_data(path)
    if pvalues is not None:
        pvalues = unpack_numpy_arrays(pvalues)
        pvalues1 = unpack_numpy_arrays(pvalues1)
        pvalues0 = unpack_numpy_arrays(pvalues0)
        af = unpack_numpy_arrays(af)
        calculate_and_print_stats(predictions, pvalues, pvalues1, pvalues0, af)