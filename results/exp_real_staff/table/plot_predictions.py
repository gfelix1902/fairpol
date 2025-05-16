import utils
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Laden der Ergebnisse
    path = utils.get_project_path() + "/results/exp_real_staff/table/"
    config_exp = utils.load_yaml("/experiments/exp_real_staff/config_real_staff")
    config_data = config_exp["data"]
    datasets = utils.load_data(config_data)
    predictions = joblib.load(path + "predictions.pkl")

    print(predictions["ols"].describe())
    print(predictions["ols"].isna().sum())  

    # Debugging: Check columns before renaming
    print("Columns before renaming:")
    print(predictions.columns)

    # Spalten umbenennen
    names_old = ["fpnet_vuf_dr_auf", "fpnet_vuf_dr_af_conf", "fpnet_vmm_dr_af_conf", "fpnet_vef_dr_af_conf", "fpnet_vef_dr_auf", "ols"]
    names_new = ["Value UF AUF", "Value UF AF", "Max-min fair AF", "Envy-free AF", "Envy-free AUF", "Ordinary Least Squares"]
    predictions = predictions.rename(columns=dict(zip(names_old, names_new)))

    # Debugging: Check columns after renaming
    print("Columns after renaming:")
    print(predictions.columns)

    # Check for NaN values in 'ols'
    print("NaN values in 'Ordinary Least Squares':", predictions["Ordinary Least Squares"].isna().sum())

    # Durchschnitt und Standardabweichung berechnen
    predictions["index"] = predictions.index
    pred_means = predictions.groupby('index').mean().drop(columns=["run"])

    # Dynamically load covariate names from YAML
    cov_names = config_data["covariate_cols"]

    # Debugging-Prints zur Überprüfung der Spaltenanzahl
    print("Shape von datasets['d_test'].data['x']:", datasets["d_test"].data["x"].shape)
    print("Anzahl der Spaltennamen in cov_names:", len(cov_names))

    # Falls notwendig, Anzahl der Kovariaten dynamisch anpassen
    cov_names = cov_names[:datasets["d_test"].data["x"].shape[1]]

    df_test = pd.DataFrame(datasets["d_test"].data["x"].detach().numpy(), columns=cov_names)
    df_test["gender"] = datasets["d_test"].data["s"][:, 1].detach().numpy()

    print("df_test['mwearn '] values:")
    print(df_test["mwearn"].describe())
    print(df_test["mwearn"].unique())

    # Durchschnittswerte mit Kovariaten kombinieren
    df_pred_mean = pd.concat([pred_means, df_test], axis=1)

    # Beispiel-Daten für Pvalues und Action Fairness (AF)
    pvalues_mean = predictions[names_new].mean()
    pvalues_mean_normalized = (pvalues_mean - pvalues_mean.min()) / (pvalues_mean.max() - pvalues_mean.min())
    pvalues_sd = predictions[names_new].std()
    af_mean = predictions[names_new].mean()  # Replace with actual AF data if available
    af_sd = predictions[names_new].std()    # Replace with actual AF data if available

    # Debugging: Check data before plotting
    print("Pvalues Mean:")
    print(pvalues_mean)
    print("Pvalues SD:")
    print(pvalues_sd)

    # Plot Pvalues Mean
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pvalues_mean_normalized.reset_index(), x="index", y=0, palette="coolwarm")
    plt.title("Pvalues Mean for Different Models")
    plt.xlabel("Model")
    plt.ylabel("Pvalues Mean")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path + "pvalues_mean_plot.pdf")
    plt.show()

    # Plot Action Fairness (AF) Mean
    plt.figure(figsize=(10, 6))
    sns.barplot(data=af_mean.reset_index(), x="index", y=0, palette="viridis")
    plt.title("Action Fairness (AF) Mean for Different Models")
    plt.xlabel("Model")
    plt.ylabel("AF Mean")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path + "af_mean_plot.pdf")
    plt.show()

    # Combine Pvalues Mean and SD into a single plot
    pvalues_combined = pd.concat([pvalues_mean, pvalues_sd], keys=["Mean", "SD"]).reset_index(level=0).rename(columns={"level_0": "Metric"})
    plt.figure(figsize=(12, 6))
    sns.barplot(data=pvalues_combined.melt(id_vars="Metric", var_name="Model", value_name="Value"), x="Model", y="Value", hue="Metric", palette="Set2")
    plt.title("Pvalues Mean and SD for Different Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path + "pvalues_combined_plot.pdf")
    plt.show()

    # Combine AF Mean and SD into a single plot
    af_combined = pd.concat([af_mean, af_sd], keys=["Mean", "SD"]).reset_index(level=0).rename(columns={"level_0": "Metric"})
    plt.figure(figsize=(12, 6))
    sns.barplot(data=af_combined.melt(id_vars="Metric", var_name="Model", value_name="Value"), x="Model", y="Value", hue="Metric", palette="Set1")
    plt.title("Action Fairness (AF) Mean and SD for Different Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path + "af_combined_plot.pdf")
    plt.show()

    # Plot Distribution of Predictions for Different Models
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=predictions[names_new], palette="coolwarm")
    plt.title("Distribution of Predictions for Different Models")
    plt.xlabel("Model")
    plt.ylabel("Prediction Values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path + "boxplot_predictions.pdf")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=predictions[names_new], palette="muted", scale="width")
    plt.title("Distribution of Predictions for Different Models")
    plt.xlabel("Model")
    plt.ylabel("Prediction Values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path + "violinplot_predictions.pdf")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pred_mean, x="mwearn", y="Ordinary Least Squares", hue="gender", palette="coolwarm")
    plt.title("OLS Predictions vs. Monthly Earnings")
    plt.xlabel("Monthly Earnings")
    plt.ylabel("OLS Predictions")
    plt.tight_layout()
    plt.savefig(path + "scatterplot_ols_vs_income.pdf")
    plt.show()

    plt.figure(figsize=(12, 8))
    corr_matrix = predictions[names_new].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Model Predictions")
    plt.tight_layout()
    plt.savefig(path + "heatmap_correlation.pdf")
    plt.show()

    # if "mwearn" not in predictions.columns:
    #     predictions = pd.concat([predictions, df_test["mwearn"]], axis=1)

    # sns.pairplot(predictions[names_new + ["mwearn"]], diag_kind="kde", palette="coolwarm")
    # plt.savefig(path + "pairplot_predictions.pdf")
    # plt.show()

    # if "index" not in predictions.columns:
    #     predictions["index"] = predictions.index

    # plt.figure(figsize=(10, 6))
    # for model in names_new:
    #     sns.lineplot(data=predictions, x="index", y=model, label=model)
    # plt.title("Predictions Across Runs")
    # plt.xlabel("Index")
    # plt.ylabel("Prediction Values")
    # plt.legend(title="Model")
    # plt.tight_layout()
    # plt.savefig(path + "lineplot_predictions.pdf")
    # plt.show()
