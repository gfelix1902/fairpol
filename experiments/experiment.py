import random
import utils
import experiments.model_training as model_training
import experiments.model_evaluation as model_evaluation
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix


def perform_experiment(config, fixed_params=None):
    config_exp = config.copy()
    runs = config_exp["experiment"]["runs"]
    initial_seed = config_exp["experiment"]["seed"]
    utils.set_seed(initial_seed)
    # Seperate data config
    config_data = config_exp["data"]
    config_exp.pop("data")
    pvalue = []
    pvalue0 = []
    pvalue1 = []
    action_fairness = []
    predictions = []
    # Start runs
    for i in range(runs):
        # Set seed
        seed = random.randint(0, 1000000)
        utils.set_seed(seed)
        # Load data
        datasets = utils.load_data(config_data)
        # Train models
        trained_models = model_training.train_models(config_exp, datasets, seed, fixed_params=fixed_params)
        # trained_models = None
        # Evaluate
        # Test predictions
        # p_predictions = model_evaluation.get_policy_predictions(trained_models, datasets["d_test"])

        if "fpnet_vuf_dm_af_gr" in trained_models.keys():
            model_test = trained_models["fpnet_vuf_dm_af_gr"]["trained_model"]
        elif "fpnet_vuf_dm_af_conf" in trained_models.keys():
            model_test = trained_models["fpnet_vuf_dm_af_conf"]["trained_model"]
        elif "fpnet_vuf_dm_af_wstein" in trained_models.keys():
            model_test = trained_models["fpnet_vuf_dm_af_wstein"]["trained_model"]
        else:
            model_test = None
        # TSNE representation plots
        if model_test is not None and config_exp["experiment"]["plotting"]:
            repr = model_test.predict_repr(datasets["d_test"][0].data)
            utils.plot_TSNE_repr_label(repr, label=datasets["d_test"][0].data["s"], title="Psi_s")
        # Calculate metrics
        policy_values = model_evaluation.get_table_pvalues(trained_models, datasets["d_test"], data_type=config_data["dataset"])
        policy_values_cond = model_evaluation.get_table_pvalues_conditional(trained_models, datasets["d_test"],
                                                                               data_type=config_data["dataset"])
        policy_cor = model_evaluation.get_table_action_fairness(trained_models, datasets["d_test"],
                                                                data_type=config_data["dataset"])
        if config_exp["experiment"]["plotting"]:
            model_evaluation.plot_policies1D(trained_models, datasets["d_test"])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print("Policy values")
            print(policy_values)
            print("Conditional policy values")
            print(policy_values_cond)
            print("Policy prediction correlation with s")
            print(policy_cor)
            # print("Action fairness of representation")
            # print(repr_cor)
            # print("Reconstruction error of representation")
            # print(repr_reconstruct)
        pvalue.append(policy_values)
        pvalue0.append(policy_values_cond.iloc[0].to_frame())
        pvalue1.append(policy_values_cond.iloc[1].to_frame())
        action_fairness.append(policy_cor)
        #Policy predictions
        df_pred = model_evaluation.get_policy_predictions(trained_models, datasets["d_test"],
                                                                   data_type=config_data["dataset"])
        predictions.append(df_pred)


    pvalue_runs = pd.concat(pvalue, axis=0)
    pvalue0_runs = pd.concat(pvalue0, axis=1).transpose()
    pvalue1_runs = pd.concat(pvalue1, axis=1).transpose()
    action_fairness_runs = pd.concat(action_fairness, axis=0)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Policy values")
        print(pvalue_runs)
        print("Policy values 0")
        print(pvalue0_runs)
        print("Policy values 1")
        print(pvalue1_runs)
        print("Action fairness")
        print(action_fairness_runs)
    #Put policy predictions into a single dataframe
    for i, df_prediction in enumerate(predictions):
        df_prediction["run"] = i
    df_predictions = pd.concat(predictions, axis=0)

    return {"pvalues": pvalue_runs, "pvalues0": pvalue0_runs, "pvalues1": pvalue1_runs, "af": action_fairness_runs,
            "predictions": df_predictions}
