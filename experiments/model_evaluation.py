import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def get_models_names(trained_models):
    models_names = []
    for model in trained_models.keys():
        models_names.append({"name": model, "model": trained_models[model]["trained_model"]})
    return models_names


def create_table(models_names, rows):
    names = [model["name"] for model in models_names]
    df_results = pd.DataFrame(columns=names, index=range(rows))
    return df_results


def get_policy_predictions(trained_models, d_test, data_type="real"):
    if data_type == "sim":
        d_test = d_test[0]
    n_test = d_test.data["y"].shape[0]
    models_names = get_models_names(trained_models)
    names = [model["name"] for model in models_names]
    df_results = pd.DataFrame(columns=names, index=range(n_test))
    for model in models_names:
        df_results.loc[:, model["name"]] = model["model"].predict(d_test).detach().numpy()
    return df_results


def get_table_pvalues(trained_models, d_test, data_type="sim"):
    models_names = get_models_names(trained_models)
    df_results = create_table(models_names, 1)
    for model in models_names:
        if data_type == "sim":
            df_results.loc[:, model["name"]] = model["model"].evaluate_policy(d_test[0])
        elif data_type == "real":
            nuisance_test = model["model"].tarnet.predict_nuisance(d_test.data)
            df_results.loc[:, model["name"]] = model["model"].evaluate_policy(d_test, oracle=False,
                                                                              nuisance=nuisance_test,
                                                                              m=model["model"].m)
    return df_results


def get_table_pvalues_conditional(trained_models, d_test, data_type="sim"):
    models_names = get_models_names(trained_models)
    if data_type == "sim":
        df_results = create_table(models_names, len(d_test))
        for model in models_names:
            # estimate conditional policy values using perturbed test data (ground truth)
            df_results.loc[:, model["name"]] = model["model"].evaluate_policy_perturbed(d_test)

    elif data_type == "real":
        df_results = create_table(models_names, 2)
        for model in models_names:
            # estimate conditional policy values on real data using estimator
            df_results.loc[:, model["name"]] = model["model"].evaluate_conditional_pvalues(d_test, oracle=False)
    return df_results


def get_table_pvalues_max(trained_models, d_test, data_type="sim"):
    models_names = get_models_names(trained_models)
    df_results = create_table(models_names, 1)
    for model in models_names:
        df_results.loc[:, model["name"]] = model["model"].evaluate_worst_case(d_test)
    return df_results


def get_table_action_fairness(trained_models, d_test, data_type="sim"):
    models_names = get_models_names(trained_models)
    df_results = create_table(models_names, 1)
    for model in models_names:
        if data_type == "sim":
            df_results.loc[:, model["name"]] = model["model"].evaluate_action_fairness(d_test)
        elif data_type == "real":
            # Policy predictions
            # Sensitive attribute
            s = np.squeeze(d_test.data["s"][:, 1].detach().numpy())
            pi_hat = np.squeeze(model["model"].predict(d_test).detach().numpy())
            test = spearmanr(a=s, b=pi_hat)
            test = test.correlation
            df_results.loc[:, model["name"]] = np.abs(test)
    return df_results


def get_table_action_fairness_repr(trained_models, d_test):
    models_names = get_models_names(trained_models)
    df_results = create_table(models_names, 1)
    for model in models_names:
        df_results.loc[:, model["name"]] = model["model"].evaluate_action_fairness_repr(d_test)
    return df_results


def get_table_reconstruction_repr(trained_models, d_test, sensitive_feat=1):
    data = d_test[0].data
    x = data["x"]
    dim_x = x.shape[1]
    x_us = x[:, 0:dim_x - sensitive_feat].detach().numpy()
    x_s = x[:, dim_x - sensitive_feat:].detach().numpy()
    models_names = get_models_names(trained_models)
    df_results = create_table(models_names, 2)
    for model in models_names:
        if hasattr(model["model"], "repr_net"):
            if model["model"].repr_net is not None:
                x_hat = model["model"].repr_net.predict_reconstruction(data).detach().numpy()
                x_hat_us = x_hat[:, 0:dim_x - sensitive_feat]
                x_hat_s = x_hat[:, dim_x - sensitive_feat:]
                df_results.loc[0, model["name"]] = np.mean(np.sqrt(np.sum((x_hat_us - x_us) ** 2, axis=1)))
                df_results.loc[1, model["name"]] = np.mean(np.sqrt(np.sum((x_hat_s - x_s) ** 2, axis=1)))
            else:
                df_results.loc[:, model["name"]] = None
        else:
            df_results.loc[:, model["name"]] = None
    return df_results


# Plot policies for binary S, dim x_us, x_s = 1
def plot_policies1D(trained_models, d_test):
    data_test = d_test[0]
    x_us_test = data_test.data["x"][:, 0:1]
    n_test = x_us_test.shape[0]
    ite_f = data_test.nuisance["mu1_f"] - data_test.nuisance["mu0_f"]
    # Plot ITEs for both sensitive groups
    ite_1 = data_test.nuisance["mu1_s1"] - data_test.nuisance["mu0"]
    ite_0 = data_test.nuisance["mu1_s0"] - data_test.nuisance["mu0"]
    ite_1 = ite_1.detach().numpy()
    ite_0 = ite_0.detach().numpy()
    ite_f = ite_f.detach().numpy()
    data = np.concatenate((x_us_test.detach().numpy(), ite_1, ite_0, ite_f), axis=1)
    models_names = get_models_names(trained_models)
    for model in models_names:
        predictions = model["model"].predict(data_test)
        data = np.concatenate((data, predictions.detach().numpy()), axis=1)
    data = data[data[:, 0].argsort()]
    plt.plot(data[:, 0], data[:, 1], label=r"$\ite1$", color="mediumblue")
    plt.plot(data[:, 0], data[:, 2], label=r"$\ite0$", color="orchid")
    plt.plot(data[:, 0], data[:, 3], label=r"$\itef$", color="lime")
    colors = ["orange", "darkred", "deepskyblue", "darkblue"]
    for i, model in enumerate(models_names):
        plt.plot(data[:, 0], data[:, 4 + i], label=model["name"], color=colors[i])
    plt.legend()
    plt.show()
