import utils
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    print("🚀 Starte das Skript...")

    path = utils.get_project_path() + "/results/exp_real_staff/table/"

    # Daten laden
    config_exp = utils.load_yaml("/experiments/exp_real_staff/config_real_staff")
    config_data = config_exp["data"]
    datasets = utils.load_data(config_data)
    predictions = joblib.load(os.path.join(path, "predictions.pkl"))

    # Spalten umbenennen für bessere Lesbarkeit
    names_old = ["fpnet_vuf_dr_auf", "fpnet_vuf_dr_af_conf", "fpnet_vmm_dr_af_conf", "fpnet_vef_dr_af_conf", "fpnet_vef_dr_auf"]
    names_new = ["Value UF AUF", "Value UF AF", "Max-min fair AF", "Envy-free AF", "Envy-free AUF"]
    predictions = predictions.rename(columns=dict(zip(names_old, names_new)))

    # Durchschnitt und Standardabweichung berechnen
    predictions["index"] = predictions.index
    pred_means = predictions.groupby('index').mean().drop(columns=["run"])
    pred_sds = predictions.groupby('index').std().drop(columns=["run"])

    # Kovariaten einfügen
    cov_names = ["age", "educ", "geddegree", "hsdegree", "mwearn", "hhsize",
                 "white_0.0", "white_1.0", "black_0.0", "black_1.0", "hispanic_0.0", "hispanic_1.0",
                 "english_0.0", "english_1.0", "cohabmarried_0.0", "cohabmarried_1.0",
                 "haschild_0.0", "haschild_1.0"]

    df_test = pd.DataFrame(datasets["d_test"].data["x"].detach().numpy(), columns=cov_names)
    df_test["gender"] = datasets["d_test"].data["s"][:, 1].detach().numpy()

    # Durchschnittswerte mit Kovariaten kombinieren
    df_pred_mean = pd.concat([pred_means, df_test], axis=1)
    df_pred_sds = pd.concat([pred_sds, df_test], axis=1)

    ### 🔥 OPTION 1: Gruppierung nach "educ" (Bildungsniveau) ###
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_pred_mean, x="educ", y="Value UF AUF", hue="gender", palette="coolwarm")
    plt.title("Vorhersage nach Bildungsniveau und Geschlecht")
    plt.xlabel("Bildungsniveau (educ)")
    plt.ylabel("Mean Prediction")
    plt.legend(title="Geschlecht", labels=["Männlich", "Weiblich"])
    plt.savefig(os.path.join(path, "plot_education.pdf"), bbox_inches='tight')
    plt.show()

    ### 🔥 OPTION 2: Korrelation zwischen Kovariaten und Vorhersagen ###
    correlation_matrix = df_pred_mean[names_new + cov_names].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Korrelation zwischen Kovariaten und Vorhersagen")
    plt.savefig(os.path.join(path, "plot_correlation.pdf"), bbox_inches='tight')
    plt.show()

    ### 🔥 OPTION 3: Streudiagramm für Einkommen vs. Bildungsniveau ###
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pred_mean, x="mwearn", y="Value UF AUF", hue="gender", style="educ", palette="deep", s=70)
    plt.title("Zusammenhang zwischen Einkommen und Vorhersagen nach Bildungsniveau")
    plt.xlabel("Monatliches Einkommen (mwearn)")
    plt.ylabel("Mean Prediction (Value UF AUF)")
    plt.legend(title="Bildungsniveau")
    plt.savefig(os.path.join(path, "plot_income.pdf"), bbox_inches='tight')
    plt.show()
