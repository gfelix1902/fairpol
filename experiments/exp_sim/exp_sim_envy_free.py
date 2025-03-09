import pandas as pd

from experiments.experiment import perform_experiment
import utils
import joblib
import numpy as np

if __name__ == "__main__":
    config_exp = utils.load_yaml("/experiments/exp_sim/config_sim")
    #Range for envy-free penalty
    #lamb_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    lamb_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.5, 2]
    #Range for action fair penalty
    gamma_range = [0, 0.05, 0.1, 0.5]
    pvalues_mean = pd.DataFrame(columns= lamb_range, index=gamma_range)
    pvalues_std = pd.DataFrame(columns= lamb_range, index=gamma_range)
    pvalues0_mean = pd.DataFrame(columns= lamb_range, index=gamma_range)
    pvalues0_std = pd.DataFrame(columns= lamb_range, index=gamma_range)
    pvalues1_mean = pd.DataFrame(columns= lamb_range, index=gamma_range)
    pvalues1_std = pd.DataFrame(columns= lamb_range, index=gamma_range)
    pvalues_diff_mean = pd.DataFrame(columns= lamb_range, index=gamma_range)
    pvalues_diff_std = pd.DataFrame(columns= lamb_range, index=gamma_range)
    af_mean = pd.DataFrame(columns= lamb_range, index=gamma_range)
    af_std = pd.DataFrame(columns= lamb_range, index=gamma_range)
    for gamma in gamma_range:
        for lamb in lamb_range:
            results = perform_experiment(config_exp, fixed_params={"lamb": lamb, "gamma": gamma})
            pvalues_mean.loc[gamma, lamb] = results["pvalues"].values.mean()
            pvalues_std.loc[gamma, lamb] = results["pvalues"].values.std()
            #Conditional policy values
            pvalues0_mean.loc[gamma, lamb] = results["pvalues0"].values.mean()
            pvalues0_std.loc[gamma, lamb] = results["pvalues0"].values.std()
            pvalues1_mean.loc[gamma, lamb] = results["pvalues1"].values.mean()
            pvalues1_std.loc[gamma, lamb] = results["pvalues1"].values.std()
            #Differences
            pvalues_diff_val = np.abs(results["pvalues1"].values - results["pvalues0"].values)
            pvalues_diff_mean.loc[gamma, lamb] = np.mean(pvalues_diff_val)
            pvalues_diff_std.loc[gamma, lamb] = np.std(pvalues_diff_val)
            #Action fairness
            af_mean.loc[gamma, lamb] = results["af"].values.mean()
            af_std.loc[gamma, lamb] = results["af"].values.std()
    # Save results to file
    path_results = utils.get_project_path() + "/results/exp_sim/envy_free/"
    joblib.dump(pvalues_mean, path_results + "pvalues_mean.pkl")
    joblib.dump(pvalues_std, path_results + "pvalues_std.pkl")
    joblib.dump(pvalues0_mean, path_results + "pvalues0_mean.pkl")
    joblib.dump(pvalues0_std, path_results + "pvalues0_std.pkl")
    joblib.dump(pvalues1_mean, path_results + "pvalues1_mean.pkl")
    joblib.dump(pvalues1_std, path_results + "pvalues1_std.pkl")
    joblib.dump(pvalues_diff_mean, path_results + "pvalues_diff_mean.pkl")
    joblib.dump(pvalues_diff_std, path_results + "pvalues_diff_std.pkl")
    joblib.dump(af_mean, path_results + "af_mean.pkl")
    joblib.dump(af_std, path_results + "af_std.pkl")
