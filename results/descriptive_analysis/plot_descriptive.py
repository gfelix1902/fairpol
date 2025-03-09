#Descriptive histograms of oregon health insurance experiment data
import random
import numpy as np
import pandas as pd
import yaml
import utils
import seaborn as sns
import matplotlib.pyplot as plt

def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

def fill_df(df_empty, data):
    df_empty.iloc[:, 0:1] = data.data["y"]
    df_empty.iloc[:, 1:2] = data.data["a"]
    df_empty.iloc[:, 2:] = np.concatenate([data.data["s"][:, 1:], data.data["x"]], axis=1)

if __name__ == "__main__":
    # Select configuration file here
    config_exp = utils.load_yaml("/experiments/exp_real/config_real")
    config_data = config_exp["data"]
    #Get datasets
    datasets = utils.load_data(config_data, standardize=False)
    n_train = datasets["d_train"].data["y"].shape[0]
    n_val = datasets["d_val"].data["y"].shape[0]
    n_test = datasets["d_test"].data["y"].shape[0]
    n = n_train + n_val + n_test
    #Create dataframe for plotting
    s_name = ["Gender"]
    x_names = ["Age", "Number of people signed up with", "Week of sign up", "Number of emergency visits", "Language"]
    df_plotting_train = pd.DataFrame(columns=["Outcome", "Treatment"] + s_name + x_names, index=range(n_train))
    fill_df(df_plotting_train, datasets["d_train"])
    df_plotting_val = pd.DataFrame(columns=["Outcome", "Treatment"] + s_name + x_names, index=range(n_val))
    fill_df(df_plotting_val, datasets["d_val"])
    df_plotting_test = pd.DataFrame(columns=["Outcome", "Treatment"] + s_name + x_names, index=range(n_test))
    fill_df(df_plotting_test, datasets["d_test"])
    df_plotting = pd.concat([df_plotting_train, df_plotting_val, df_plotting_test], ignore_index=True, sort=False)



    #df_plotting = df_plotting.melt(var_name="Variable", value_name="Value")
    sns.set_theme(style="darkgrid")
    #grid = sns.FacetGrid(df_plotting, col="Variable", col_wrap=4, height=3)

    # define plotting region (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    # create boxplot in each subplot
    sns.histplot(data=df_plotting, x='Outcome', ax=axes[0, 0], stat="frequency", discrete=False).set(yticklabels=[])
    sns.histplot(data=df_plotting, x='Treatment', ax=axes[0, 1], stat="frequency", discrete=True).set(yticklabels=[], xticks=[0, 1], xticklabels=["0", "1"])
    sns.histplot(data=df_plotting, x=s_name[0], ax=axes[0, 2], stat="frequency", discrete=True).set(yticklabels=[], xticks=[0, 1], xticklabels=["Male", "Female"])
    sns.histplot(data=df_plotting, x=x_names[0], ax=axes[0, 3], stat="frequency", discrete=True).set(yticklabels=[])
    sns.histplot(data=df_plotting, x=x_names[1], ax=axes[1, 0], stat="frequency", discrete=True).set(yticklabels=[], xticks=[1, 2, 3], xticklabels=["1", "2", "3"])
    sns.histplot(data=df_plotting, x=x_names[2], ax=axes[1, 1], stat="frequency", discrete=True).set(yticklabels=[])
    sns.histplot(data=df_plotting, x=x_names[3], ax=axes[1, 2], stat="frequency", discrete=True).set(yticklabels=[])
    sns.histplot(data=df_plotting, x=x_names[4], ax=axes[1, 3], stat="frequency", discrete=True).set(yticklabels=[], xticks=[0, 1], xticklabels=["Other", "English"])

    #grid.add_legend()
    plt.savefig(utils.get_project_path() + "/results/descriptive_analysis/plot_descriptive.pdf", bbox_inches='tight')
    plt.show()
