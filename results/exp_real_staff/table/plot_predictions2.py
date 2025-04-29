import utils
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 🔧 Matplotlib-Stil einstellen
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

if __name__ == "__main__":
    # 🔹 Laden der Ergebnisse
    path = utils.get_project_path() + "/results/exp_real_staff/table/"
    config_exp = utils.load_yaml("/experiments/exp_real_staff/config_real_staff")
    config_data = config_exp["data"]
    datasets = utils.load_data(config_data)
    predictions = joblib.load(path + "predictions.pkl")

    # 🔹 Spalten umbenennen
    names_old = ["fpnet_vuf_dr_auf", "fpnet_vuf_dr_af_conf", "fpnet_vmm_dr_af_conf", "fpnet_vef_dr_af_conf", "fpnet_vef_dr_auf"]
    names_new = ["Value UF AUF", "Value UF AF", "Max-min fair AF", "Envy-free AF", "Envy-free AUF"]
    predictions = predictions.rename(columns=dict(zip(names_old, names_new)))

    # 🔹 Durchschnitt und Standardabweichung berechnen
    predictions["index"] = predictions.index
    pred_means = predictions.groupby('index').mean().drop(columns=["run"])

    # 🔹 Kovariaten einfügen
    cov_names = ["age", "educ", "geddegree", "hsdegree", "mwearn", "hhsize",
                 "white_0.0", "white_1.0", "black_0.0", "black_1.0", "hispanic_0.0", "hispanic_1.0",
                 "english_0.0", "english_1.0", "cohabmarried_0.0", "cohabmarried_1.0",
                 "haschild_0.0", "haschild_1.0"]

    df_test = pd.DataFrame(datasets["d_test"].data["x"].detach().numpy(), columns=cov_names)
    df_test["gender"] = datasets["d_test"].data["s"][:, 1].detach().numpy()

    # 🔹 Durchschnittswerte mit Kovariaten kombinieren
    df_pred_mean = pd.concat([pred_means, df_test], axis=1)

    # === 🔥 1. Verbesserter Balkendiagramm-Plot ===
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_pred_mean, x="educ", y="Value UF AUF", hue="gender",
                palette="viridis", ci="sd", errorbar="sd")
    
    plt.title("Durchschnittliche Vorhersage nach Bildungsniveau und Geschlecht", fontsize=16)
    plt.xlabel("Bildungsniveau", fontsize=14)
    plt.ylabel("Durchschnittliche Vorhersage (Value UF AUF)", fontsize=14)
    plt.legend(title="Geschlecht", labels=["Männlich", "Weiblich"], title_fontsize=12, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.savefig(path + "plot_education_improved.pdf", bbox_inches="tight")
    plt.show()

    # === 🔥 2. Verbesserter PCA-Plot ===
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df_pred_mean[names_new + cov_names])
    pca_df = pd.DataFrame(pca_data, columns=["PCA 1", "PCA 2"])
    pca_df["gender"] = df_pred_mean["gender"]

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x="PCA 1", y="PCA 2", hue="gender",
                    palette="coolwarm", s=100, edgecolor="black", alpha=0.75)

    plt.title("PCA-Plot für Kovariaten und Vorhersagen", fontsize=16)
    plt.xlabel("PCA 1", fontsize=14)
    plt.ylabel("PCA 2", fontsize=14)
    plt.legend(title="Geschlecht", labels=["Männlich", "Weiblich"], title_fontsize=12, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(path + "plot_pca.pdf", bbox_inches="tight")
    plt.show()

    # === 🔥 3. Verbesserter Hexbin-Plot ===
    plt.figure(figsize=(12, 7))
    hb = plt.hexbin(df_pred_mean["mwearn"], df_pred_mean["Value UF AUF"],
                    gridsize=40, cmap="viridis", mincnt=1, linewidths=0.3)

    plt.colorbar(hb, label="Anzahl Punkte")
    plt.title("Einkommen vs. Vorhersage (Value UF AUF)", fontsize=16)
    plt.xlabel("Monatliches Einkommen", fontsize=14)
    plt.ylabel("Vorhersagewert (Value UF AUF)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(path + "plot_income_improved.pdf", bbox_inches="tight")
    plt.show()
