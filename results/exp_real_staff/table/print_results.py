import utils
import pandas as pd
import numpy as np
import joblib

def print_df(title, df):
    print(f"\n{'-'*40}\n{title}\n{'-'*40}")
    print(df.round(4))

if __name__ == "__main__":
    path = utils.get_project_path() + "/results/exp_real_staff/table/"

    # Load data
    config_exp = utils.load_yaml("/experiments/exp_real_staff/config_real_staff")
    config_data = config_exp["data"]
    datasets = utils.load_data(config_data)
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

    # Formattierte Ausgabe
    print_df("Pvalues Mean", pvalues_means)
    print_df("Pvalues SD", pvalues_sds)
    print_df("Pvalues 1 Mean", pvalues1_means)
    print_df("Pvalues 1 SD", pvalues1_sds)
    print_df("Pvalues 0 Mean", pvalues0_means)
    print_df("Pvalues 0 SD", pvalues0_sds)
    print_df("AF Mean", af_means)
    print_df("AF SD", af_sds)
    print_df("Predictions Mean", pred_means)
    print_df("Predictions SD", pred_sds)

    # Plot OLS predictions if present
    if "ols" in pred_means.columns:
        import matplotlib.pyplot as plt

        # True values from test set
        y_true = datasets["d_test"].data["y"].cpu().numpy().ravel()
        # Align indices if necessary
        y_pred = pred_means["ols"].values
        if len(y_true) != len(y_pred):
            print("Warning: Length mismatch between y_true and y_pred. Plot may be incorrect.")

        # 1. True vs. Predicted
        plt.figure(figsize=(7, 7))
        plt.scatter(y_true, y_pred, alpha=0.5, label="Samples")
        # Regression line
        m, b = np.polyfit(y_true, y_pred, 1)
        plt.plot(y_true, m*y_true + b, color='blue', label='Fit')
        # Diagonal
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("OLS: True vs. Predicted")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 2. Residuals
        residuals = y_true - y_pred
        plt.figure(figsize=(7, 4))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("OLS: Residual Plot")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 3. Residuals Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=30, alpha=0.7, color='gray')
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title("OLS: Residuals Distribution")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
