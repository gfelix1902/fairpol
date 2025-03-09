import utils
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

""" increase white margins of plot"""
def increase_margins(ax, top=0.05, bottom=0.05, left=0.05, right=0.05):
    fig = ax.get_figure()
    fig.subplots_adjust(top=1 - top, bottom=bottom, left=left, right=1 - right)
    return ax

if __name__ == "__main__":
    path = utils.get_project_path() + "/results/exp_sim/envy_free/"
    # Load results
    pvalues_mean = joblib.load(path + "pvalues_mean.pkl")
    pvalues_std = joblib.load(path + "pvalues_std.pkl")
    # Conditional policy values
    pvalues0_mean = joblib.load(path + "pvalues0_mean.pkl")
    pvalues0_std = joblib.load(path + "pvalues0_std.pkl")
    pvalues1_mean = joblib.load(path + "pvalues1_mean.pkl")
    pvalues1_std = joblib.load(path + "pvalues1_std.pkl")
    # Difference
    pvalues_diff_mean = joblib.load(path + "pvalues_diff_mean.pkl")
    pvalues_diff_std = joblib.load(path + "pvalues_diff_std.pkl")
    # Action fairness
    af_mean = joblib.load(path + "af_mean.pkl")
    af_std = joblib.load(path + "af_std.pkl")
    lamb_range = pvalues_mean.columns
    gamma_range = pvalues_mean.index
    metrics = ["Policy value", "Policy value S=0", "Policy value S=1"]

    data_plot = pd.DataFrame(columns=["gamma", "lambda", "metric", "mean", "ci_lower", "ci_upper"])
    for gamma in gamma_range:
        for lamb in lamb_range:
            data_chunk = pd.DataFrame(columns=["gamma", "lambda", "metric", "mean", "ci_lower", "ci_upper"],
                                      index=range(3))
            data_chunk.iloc[0] = [gamma, lamb, metrics[0], pvalues_mean.loc[gamma, lamb],
                                  pvalues_mean.loc[gamma, lamb] - pvalues_std.loc[gamma, lamb],
                                  pvalues_mean.loc[gamma, lamb] + pvalues_std.loc[gamma, lamb]]
            data_chunk.iloc[1] = [gamma, lamb, metrics[1], pvalues0_mean.loc[gamma, lamb],
                                  pvalues0_mean.loc[gamma, lamb] - pvalues0_std.loc[gamma, lamb],
                                  pvalues0_mean.loc[gamma, lamb] + pvalues0_std.loc[gamma, lamb]]
            data_chunk.iloc[2] = [gamma, lamb, metrics[2], pvalues1_mean.loc[gamma, lamb],
                                  pvalues1_mean.loc[gamma, lamb] - pvalues1_std.loc[gamma, lamb],
                                  pvalues1_mean.loc[gamma, lamb] + pvalues1_std.loc[gamma, lamb]]
            # data_chunk.iloc[3] = [gamma, lamb, metrics[3], af_mean.loc[gamma, lamb],
            #                      af_mean.loc[gamma, lamb] - af_std.loc[gamma, lamb], af_mean.loc[gamma, lamb] + af_std.loc[gamma, lamb]]
            data_plot = pd.concat([data_plot, data_chunk])

    data_plot = data_plot.reset_index().drop(columns=["index"])
    df_plot_lambda = data_plot[data_plot.gamma == 0.1]
    # df_plot_lambda = df_plot_lambda[df_plot_lambda.metric != "Action fairness"]
    colors = ["darkred", "darkgreen", "darkblue"]
    # metrics = ["Policy value", "Worst case policy value", "Policy value difference"]

    # df_plot_lambda is a dataframe with columns gamma, lambda, metric, mean, ci_lower, ci_upper
    # Create plot using df_plot_lambda
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    ax = sns.lineplot(data=df_plot_lambda, x="lambda", y="mean", hue="metric", palette=colors)
    # Create confidence intervals in ax using ci_lower and ci_upper from df_plot_lambda as standard deviation
    for j, metric in enumerate(metrics):
        data = df_plot_lambda[df_plot_lambda.metric == metrics[j]]
        y1 = data.ci_upper.to_numpy()
        y1 = [float(y) for y in y1]
        y2 = data.ci_lower.to_numpy()
        y2 = [float(y) for y in y2]
        ax.fill_between(lamb_range, y1, y2, alpha=0.1, color=colors[j])

    # set labels and title
    ax.set_xlabel(r'Envy-freeness parameter $\lambda$', fontsize=20)
    ax.set_ylabel("Policy value")
    # add legend11^
    ax.legend(title="", loc="upper right", labels=["Overall", "$\it{S}$=0", "$\it{S}$=1"])
    ax = increase_margins(ax, top=0.1, bottom=0.15, left=0.2, right=0.1)

    plt.savefig(path + "plot_envy_free_lambda.pdf")
    plt.show()


    # Filter data_plot for all entries with lambda = 0.1
    df_plot_gamma = data_plot[data_plot["lambda"] == 0.6]
    ax = sns.lineplot(data=df_plot_gamma, x="gamma", y="mean", hue="metric", palette=colors)
    # Create confidence intervals in ax using ci_lower and ci_upper from df_plot_lambda as standard deviation
    for j, metric in enumerate(metrics):
        data = df_plot_gamma[df_plot_gamma.metric == metrics[j]]
        y1 = data.ci_upper.to_numpy()
        y1 = [float(y) for y in y1]
        y2 = data.ci_lower.to_numpy()
        y2 = [float(y) for y in y2]
        ax.fill_between(gamma_range, y1, y2, alpha=0.1, color=colors[j])

    # set labels and title
    ax.set_xlabel(r'Regularization parameter $\gamma$')
    ax.set_ylabel("Policy value")
    # add legend
    ax.legend(title="", loc="upper right", labels=["Overall", "$\it{S}$=0", "$\it{S}$=1"])
    ax.set_ylim(0, 0.15)
    ax = increase_margins(ax, top=0.1, bottom=0.15, left=0.2, right=0.1)
    plt.savefig(path + "plot_envy_free_gamma.pdf")
    plt.show()
    # grid = sns.FacetGrid(data_plot, col="metric", hue="gamma", col_wrap=2, height=3, palette=colors, sharey=False, sharex=True)
    # grid.map(plt.plot, "lambda", "mean")
    # grid.set(xlabel=r'$\lambda$', ylabel="Value")
    # axes = grid.axes.flatten()
    # axes[0].set_title(metrics[0])
    # axes[1].set_title(metrics[1])
    # axes[2].set_title(metrics[2])
    # axes[3].set_title(metrics[3])
    # grid.add_legend(title=r'$\gamma$')

    # for i, ax in enumerate(grid.axes):
    #    for j, gamma in enumerate(gamma_range):
    #        data = data_plot[data_plot.gamma == gamma]
    #        test = data.metric == metrics[i]
    #        data = data[data.metric == metrics[i]]
    #        y1 = data.ci_upper.to_numpy()
    #        y1 = [float(y) for y in y1]
    #        y2 = data.ci_lower.to_numpy()
    #        y2 = [float(y) for y in y2]
    #        ax.fill_between(lamb_range, y1, y2, alpha=0.1, color=colors[j])
