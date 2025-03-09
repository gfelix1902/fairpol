import utils
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = utils.get_project_path() + "/results/exp_real/table/"

    # Load data
    config_exp = utils.load_yaml("/experiments/exp_real/config_real")
    config_data = config_exp["data"]
    datasets = utils.load_data(config_data)
    predictions = []
    predictions = joblib.load(path + "predictions.pkl")
    #Rename columns
    names_old = ["fpnet_vuf_dr_auf", "fpnet_vuf_dr_af_conf", "fpnet_vmm_dr_af_conf", "fpnet_vef_dr_af_conf", "fpnet_vef_dr_auf"]
    names_new = ["Value UF AUF", "Value UF AF", "Max-min fair AF", "Envy-free AF", "Envy-free AUF"]
    predictions = predictions.rename(columns={names_old[0]: names_new[0], names_old[1]: names_new[1],
                                              names_old[2]: names_new[2], names_old[3]: names_new[3],
                                              names_old[4]: names_new[4]})
    pred_means = []
    pred_sds = []

    # Predictions
    predictions["index"] = predictions.index
    pred_means = predictions.groupby('index').mean().drop(columns=["run"])
    pred_sds = predictions.groupby('index').std().drop(columns=["run"])
    cov_names = ["Age", "num_signed_up", "week_signup", "Number of emergency visits", "language"]
    df_test = pd.DataFrame(datasets["d_test"].data["x"].detach().numpy(), columns=cov_names)
    df_test["gender"] = datasets["d_test"].data["s"][:, 1].detach().numpy()
    df_pred_mean = pd.concat([pred_means, df_test], axis=1)
    df_pred_sds = pd.concat([pred_sds, df_test], axis=1)

    #Data for plotting mean predictionss over age
    df_pred_mean_age = df_pred_mean.groupby("Age").mean()
    df_pred_mean_age = df_pred_mean_age.drop(
        columns=["num_signed_up", "week_signup", "Number of emergency visits", "language", "gender"], axis=1)
    df_pred_sds_age = df_pred_mean.groupby("Age").std()
    df_pred_sds_age = df_pred_sds_age.drop(
        columns=["num_signed_up", "week_signup", "Number of emergency visits", "language", "gender"], axis=1)
    mean_age = datasets["d_test"].scaling_params["x"]["m"][0]
    sd_age = datasets["d_test"].scaling_params["x"]["sd"][0]
    df_pred_mean_age["Age"] = df_pred_mean_age.index * sd_age + mean_age
    df_pred_sds_age["Age"] = df_pred_sds_age.index * sd_age + mean_age
    df_mean_age_melted = pd.melt(df_pred_mean_age, value_vars=names_new,
                                   var_name='Method', value_name='mean', id_vars="Age")
    df_sds_age_melted = pd.melt(df_pred_sds_age, value_vars=names_new,
                                   var_name='Method', value_name='sd', id_vars="Age")
    df_plot_age = df_mean_age_melted.merge(right=df_sds_age_melted, on=["Method", "Age"], how="inner")
    df_plot_age = pd.melt(df_plot_age, value_vars="Age",
                                   var_name='Covariate', value_name='x', id_vars=["mean", "sd", "Method"])

    #Data for plotting mean predictionss over age
    df_pred_mean_visit = df_pred_mean.groupby("Number of emergency visits").mean()
    df_pred_mean_visit = df_pred_mean_visit.drop(
        columns=["num_signed_up", "week_signup", "Age", "language", "gender"], axis=1)
    df_pred_sds_visit = df_pred_mean.groupby("Number of emergency visits").std()
    df_pred_sds_visit = df_pred_sds_visit.drop(
        columns=["num_signed_up", "week_signup", "Age", "language", "gender"], axis=1)
    mean_visit = datasets["d_test"].scaling_params["x"]["m"][3]
    sd_visit = datasets["d_test"].scaling_params["x"]["sd"][3]
    df_pred_mean_visit["Number of emergency visits"] = df_pred_mean_visit.index * sd_visit + mean_visit
    df_pred_sds_visit["Number of emergency visits"] = df_pred_sds_visit.index * sd_visit + mean_visit
    df_pred_sds_visit = df_pred_sds_visit.fillna(df_pred_sds_visit.median())
    df_mean_visit_melted = pd.melt(df_pred_mean_visit, value_vars=names_new,
                                   var_name='Method', value_name='mean', id_vars="Number of emergency visits")
    df_sds_visit_melted = pd.melt(df_pred_sds_visit, value_vars=names_new,
                                   var_name='Method', value_name='sd', id_vars="Number of emergency visits")
    df_plot_visit = df_mean_visit_melted.merge(right=df_sds_visit_melted, on=["Method", "Number of emergency visits"], how="inner")
    df_plot_visit = pd.melt(df_plot_visit, value_vars="Number of emergency visits",
                                   var_name='Covariate', value_name='x', id_vars=["mean", "sd", "Method"])
    #Final dataframe for plotting
    df_plot = pd.concat([df_plot_age, df_plot_visit], axis=0)


    #Plotting
    colors = ["darkred", "deepskyblue", "darkgreen", "darkblue", "orange"]
    metrics = ["Age", "Number of emergency visits"]
    grid = sns.FacetGrid(df_plot, col="Covariate", hue="Method", col_wrap=2, height=3, palette=colors, sharey=False,
                         sharex=False)
    grid.map(plt.plot, "x", "mean")
    grid.set(xlabel="Age", ylabel="Mean prediction")
    axes = grid.axes.flatten()
    axes[0].set_title(metrics[0])
    axes[0].set_xlim(20, 60)
    axes[1].set_title(metrics[1])
    axes[1].set_xlabel("Visits")
    axes[1].set_xlim(0, 20)
    grid.add_legend(title='Method')

    for i, ax in enumerate(grid.axes):
        for j, method in enumerate(names_new):
            data = df_plot[df_plot.Method == method]
            data = data[data.Covariate == metrics[i]]
            y1 = data["mean"].to_numpy() + data["sd"].to_numpy()
            y1 = [float(y) for y in y1]
            y2 = data["mean"].to_numpy() - data["sd"].to_numpy()
            y2 = [float(y) for y in y2]
            ax.fill_between(data.x, y1, y2, alpha=0.1, color=colors[j])

    plt.savefig(path + "plot_pred.pdf", bbox_inches='tight')
    plt.show()