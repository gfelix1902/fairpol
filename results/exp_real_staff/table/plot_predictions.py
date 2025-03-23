import utils
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Das Skript wird ausgeführt!")

    path = utils.get_project_path() + "/results/exp_real_staff/table/"

    # Load data
    config_exp = utils.load_yaml("/experiments/exp_real_staff/config_real_staff")
    config_data = config_exp["data"]
    datasets = utils.load_data(config_data)
    predictions = joblib.load(path + "predictions.pkl")

    # Rename columns
    names_old = ["fpnet_vuf_dr_auf", "fpnet_vuf_dr_af_conf", "fpnet_vmm_dr_af_conf", "fpnet_vef_dr_af_conf", "fpnet_vef_dr_auf"]
    names_new = ["Value UF AUF", "Value UF AF", "Max-min fair AF", "Envy-free AF", "Envy-free AUF"]
    predictions = predictions.rename(columns={names_old[0]: names_new[0], names_old[1]: names_new[1],
                                              names_old[2]: names_new[2], names_old[3]: names_new[3],
                                              names_old[4]: names_new[4]})

    # Calculate means and stds
    predictions["index"] = predictions.index
    pred_means = predictions.groupby('index').mean().drop(columns=["run"])
    pred_sds = predictions.groupby('index').std().drop(columns=["run"])

    # Load covariate data
    cov_names = ["age", "educ", "geddegree", "hsdegree", "mwearn", "hhsize",
           "white_0.0", "white_1.0", "black_0.0", "black_1.0", "hispanic_0.0", "hispanic_1.0",
           "english_0.0", "english_1.0", "cohabmarried_0.0", "cohabmarried_1.0",
           "haschild_0.0", "haschild_1.0"]  # x_names verwenden
    df_test = pd.DataFrame(datasets["d_test"].data["x"].detach().numpy(), columns=cov_names)
    df_test["gender"] = datasets["d_test"].data["s"][:, 1].detach().numpy()
    df_pred_mean = pd.concat([pred_means, df_test], axis=1)
    df_pred_sds = pd.concat([pred_sds, df_test], axis=1)

    # Prepare for plotting age-based predictions
    df_pred_mean_age = df_pred_mean.groupby("age").mean()  # nur nach "age" gruppieren
    df_pred_sds_age = df_pred_mean.groupby("age").std()    # nach "age" gruppieren

    # Apply scaling
    mean_age = datasets["d_test"].scaling_params["x"]["m"][0]
    sd_age = datasets["d_test"].scaling_params["x"]["sd"][0]
    df_pred_mean_age["age"] = df_pred_mean_age.index * sd_age + mean_age
    df_pred_sds_age["age"] = df_pred_sds_age.index * sd_age + mean_age

    # Melt for plotting
    df_mean_age_melted = pd.melt(df_pred_mean_age, value_vars=names_new, var_name='Method', value_name='mean', id_vars="age")
    df_sds_age_melted = pd.melt(df_pred_sds_age, value_vars=names_new, var_name='Method', value_name='sd', id_vars="age")
    df_plot_age = df_mean_age_melted.merge(right=df_sds_age_melted, on=["Method", "age"], how="inner")
    
    # Plotting
    colors = ["darkred", "deepskyblue", "darkgreen", "darkblue", "orange"]
    grid = sns.FacetGrid(df_plot_age, col="age", hue="Method", col_wrap=2, height=3, palette=colors, sharey=False, sharex=False)
    grid.map(plt.plot, "age", "mean")
    grid.set(xlabel="Age", ylabel="Mean prediction")
    grid.add_legend(title='Method')

    # Fill between mean ± sd
    for i, ax in enumerate(grid.axes):
        for j, method in enumerate(names_new):
            data = df_plot_age[df_plot_age.Method == method].copy() # copy um die originalen Daten nicht zu verändern.
            data["mean"] = pd.to_numeric(data["mean"], errors='coerce')
            data["sd"] = pd.to_numeric(data["sd"], errors='coerce')
            data = data.dropna(subset=["mean", "sd"]) # Entferne NaN werte
            y1 = data["mean"].to_numpy() + data["sd"].to_numpy()
            y2 = data["mean"].to_numpy() - data["sd"].to_numpy()
            ax.fill_between(data["age"], y1, y2, alpha=0.1, color=colors[j])

    plt.savefig(path + "plot_pred.pdf", bbox_inches='tight')
    plt.show()