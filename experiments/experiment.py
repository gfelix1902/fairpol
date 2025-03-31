import random
import utils
import experiments.model_training as model_training
import experiments.model_evaluation as model_evaluation
import pandas as pd
import numpy as np
import os

def perform_experiment(config, fixed_params=None):
    """
    Führt ein Experiment mit mehreren Durchläufen durch und gibt die Ergebnisse zurück.
    """
    config_exp = config.copy()
    runs = config_exp["experiment"]["runs"]
    initial_seed = config_exp["experiment"]["seed"]
    utils.set_seed(initial_seed)
    config_data = config_exp.pop("data")

    pvalue_list = []
    pvalue0_list = []
    pvalue1_list = []
    action_fairness_list = []
    predictions_list = []

    for run_index in range(runs):
        seed = random.randint(0, 1000000)
        utils.set_seed(seed)

        try:
            datasets = utils.load_data(config_data)
            trained_models = model_training.train_models(config_exp, datasets, seed, fixed_params=fixed_params)
            policy_values = model_evaluation.get_table_pvalues(trained_models, datasets["d_test"], data_type=config_data["dataset"])
            policy_values_cond = model_evaluation.get_table_pvalues_conditional(trained_models, datasets["d_test"], data_type=config_data["dataset"])
            policy_cor = model_evaluation.get_table_action_fairness(trained_models, datasets["d_test"], data_type=config_data["dataset"])
            df_pred = model_evaluation.get_policy_predictions(trained_models, datasets["d_test"], data_type=config_data["dataset"])
        except Exception as e:
            print(f"Error during run {run_index}: {e}")
            continue

        if policy_values is None or policy_values_cond is None or policy_cor is None or df_pred is None:
            print(f"Warning: No valid results from evaluation in run {run_index}.")
            continue

        pvalue_list.append(policy_values)
        pvalue0_list.append(policy_values_cond.iloc[0].to_frame())
        pvalue1_list.append(policy_values_cond.iloc[1].to_frame())
        action_fairness_list.append(policy_cor)
        df_pred["run"] = run_index
        predictions_list.append(df_pred)

        if config_exp["experiment"]["plotting"]:
            model_test = None
            if "fpnet_vuf_dm_af_gr" in trained_models:
                model_test = trained_models["fpnet_vuf_dm_af_gr"]["trained_model"]
            elif "fpnet_vuf_dm_af_conf" in trained_models:
                model_test = trained_models["fpnet_vuf_dm_af_conf"]["trained_model"]
            elif "fpnet_vuf_dm_af_wstein" in trained_models:
                model_test = trained_models["fpnet_vuf_dm_af_wstein"]["trained_model"]

            if model_test is not None:
                repr = model_test.predict_repr(datasets["d_test"][0].data)
                utils.plot_TSNE_repr_label(repr, label=datasets["d_test"][0].data["s"], title="Psi_s")
            else:
                print(f"Warning: No suitable model for TSNE plotting in run {run_index}.")

    pvalue_runs = pd.concat(pvalue_list, axis=0)
    pvalue0_runs = pd.concat(pvalue0_list, axis=1).transpose()
    pvalue1_runs = pd.concat(pvalue1_list, axis=1).transpose()
    action_fairness_runs = pd.concat(action_fairness_list, axis=0)
    df_predictions = pd.concat(predictions_list, axis=0)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Policy values\n", pvalue_runs)
        print("Policy values 0\n", pvalue0_runs)
        print("Policy values 1\n", pvalue1_runs)
        print("Action fairness\n", action_fairness_runs)

    return {"pvalues": pvalue_runs, "pvalues0": pvalue0_runs, "pvalues1": pvalue1_runs, "af": action_fairness_runs, "predictions": df_predictions}