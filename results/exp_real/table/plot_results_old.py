import utils
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = utils.get_project_path() + "/results/exp_real/table/"

    #Load data
    config_exp = utils.load_yaml("/experiments/exp_real/config_real")
    config_data = config_exp["data"]
    datasets = utils.load_data(config_data)
    #Delta
    delta_list = [0.05, 0.1, 0.15, 0.175, 0.185, 0.2]
    predictions = []
    pvalues = []
    pvalues1 = []
    for delta in delta_list:
        predictions.append(joblib.load(path + "predictions_" + str(delta) + ".pkl"))
        pvalues.append(joblib.load(path + "pvalues_" + str(delta) + ".pkl"))
        pvalues1.append(joblib.load(path + "pvalues1_" + str(delta) + ".pkl"))

    pvalues_means = []
    pvalues_sds = []
    pvalues1_means = []
    pvalues1_sds = []
    pred_means = []
    pred_sds = []
    budget_means = []
    budget_sds = []
    for i, pred in enumerate(predictions):
        #Policy values
        pvalues_means.append(pvalues[i].mean().to_frame().transpose())
        pvalues_sds.append(pvalues[i].std().to_frame().transpose())
        pvalues1_means.append(pvalues1[i].mean().to_frame().transpose())
        pvalues1_sds.append(pvalues1[i].std().to_frame().transpose())
        #Budget
        budget = pred.groupby('run').mean()
        budget_means.append(budget.mean().to_frame().transpose())
        budget_sds.append(budget.std().to_frame().transpose())
        #Predictions
        pred["index"] = pred.index
        pred_means.append(pred.groupby('index').mean().drop(columns=["run"]))
        pred_sds.append(pred.groupby('index').std().drop(columns=["run"]))

        pvalues_means[i]["delta"] = delta_list[i]
        pvalues_sds[i]["delta"] = delta_list[i]
        pvalues1_means[i]["delta"] = delta_list[i]
        pvalues1_sds[i]["delta"] = delta_list[i]
        budget_means[i]["delta"] = delta_list[i]
        budget_sds[i]["delta"] = delta_list[i]
        pred_means[i]["delta"] = delta_list[i]
        pred_sds[i]["delta"] = delta_list[i]
    pvalues_means = pd.concat(pvalues_means)
    pvalues_sds = pd.concat(pvalues_sds)
    pvalues1_means = pd.concat(pvalues1_means)
    pvalues1_sds = pd.concat(pvalues1_sds)
    budget_means = pd.concat(budget_means)
    budget_sds = pd.concat(budget_sds)
    pred_means = pd.concat(pred_means)
    pred_sds = pd.concat(pred_sds)

    #Create dataframe for plotting
    #Budget
    budget_means_melted = pd.melt(budget_means, value_vars=['fpnet_vuf_dr_af_conf', 'fpnet_vmm_dr_af_conf', 'fpnet_vef_dr_af_conf'],
                                   var_name='Method', value_name='budget', id_vars=["delta"])
    #Pvalue
    pvalues_means_melted = pd.melt(pvalues_means, value_vars=['fpnet_vuf_dr_af_conf', 'fpnet_vmm_dr_af_conf', 'fpnet_vef_dr_af_conf'],
                                   var_name='Method', value_name='mean', id_vars=["delta"])
    pvalues_sds_melted = pd.melt(pvalues_sds, value_vars=['fpnet_vuf_dr_af_conf', 'fpnet_vmm_dr_af_conf', 'fpnet_vef_dr_af_conf'],
                                   var_name='Method', value_name='sd', id_vars=["delta"])
    pvalues_melted = pvalues_means_melted.merge(right=pvalues_sds_melted, on=["delta", "Method"], how="inner")
    pvalues_melted = pvalues_melted.merge(right=budget_means_melted, on=["delta", "Method"], how="inner")
    pvalues_melted["metric"] = "pvalue"

    pvalues1_means_melted = pd.melt(pvalues1_means, value_vars=['fpnet_vuf_dr_af_conf', 'fpnet_vmm_dr_af_conf', 'fpnet_vef_dr_af_conf'],
                                   var_name='Method', value_name='mean', id_vars=["delta"])
    pvalues1_sds_melted = pd.melt(pvalues1_sds, value_vars=['fpnet_vuf_dr_af_conf', 'fpnet_vmm_dr_af_conf', 'fpnet_vef_dr_af_conf'],
                                   var_name='Method', value_name='sd', id_vars=["delta"])
    pvalues1_melted = pvalues1_means_melted.merge(right=pvalues1_sds_melted, on=["delta", "Method"], how="inner")
    pvalues1_melted = pvalues1_melted.merge(right=budget_means_melted, on=["delta", "Method"], how="inner")
    pvalues1_melted["metric"] = "pvalue1"

    df_plot = pd.concat([pvalues_melted, pvalues1_melted])

    #Plotting
    colors = ["darkred", "deepskyblue", "darkgreen", "darkblue"]
    metrics = ["Policy value", "Worst case policy value"]
    grid = sns.FacetGrid(df_plot, col="metric", hue="Method", col_wrap=2, height=3, palette=colors, sharey=False,
                         sharex=True)
    grid.map(plt.plot, "budget", "mean")
    grid.set(xlabel=r'$\lambda$', ylabel="Value")
    axes = grid.axes.flatten()
    axes[0].set_title(metrics[0])
    axes[1].set_title(metrics[1])
    #axes[2].set_title(metrics[2])
    #daxes[3].set_title(metrics[3])
    grid.add_legend(title='Method')

    plt.savefig(path + "plot_budget.pdf", bbox_inches='tight')
    plt.show()