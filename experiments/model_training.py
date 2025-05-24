import utils
import models.fp_net as fp
import models.unfair_baselines as ub
import numpy as np 
from models.ols_model import OLSModel
import pandas as pd # Add this import if not already present


def train_models(config_exp, datasets, seed=1, fixed_params=None):
    hyper_path = config_exp["experiment"]["hyper_path"]
    model_configs = config_exp["experiment"]["models"]
    config_af = utils.get_config_af(model_configs)
    nuisance = {}

    # Train nuisance models
    # print("Nuisance training----------------------------------")

    if not config_exp["experiment"]["oracle_nuisance"]:
        # Nuisance model TARNet
        # print("Train nuisance TARNet")
        utils.set_seed(seed)
        # print(f"Hyperparameter-Pfad: {hyper_path + '/nuisance/tarnet'}")
        config_tarnet = get_hyper(config_exp, datasets["d_train"], hyper_path + "/nuisance/tarnet")
        # print(f"TARNet config: {config_tarnet}")  # Log the configuration
        tarnet = ub.train_tarnet(datasets, config_tarnet)
        nuisance["tarnet"] = tarnet["trained_model"]
        tarnet_pred = nuisance["tarnet"].predict_nuisance(datasets["d_test"].data)

        # Log the predictions to check for NaN values
        tarnet_predm1 = tarnet_pred["mu1"].detach().numpy()
        tarnet_predm0 = tarnet_pred["mu0"].detach().numpy()
        tarnet_predp = tarnet_pred["prop"].detach().numpy()

        # print(f"TARNet predictions: mu1: {tarnet_predm1[:5]}, mu0: {tarnet_predm0[:5]}, prop: {tarnet_predp[:5]}")  # Log first 5 predictions

        # Check for NaN values in the predictions
        if np.isnan(tarnet_predm1).any() or np.isnan(tarnet_predm0).any() or np.isnan(tarnet_predp).any():
            print("Warning: NaN values detected in TARNet predictions!")

        pred_ite = tarnet_predm1 - tarnet_predm0
        # print(f"ITE predictions: {pred_ite[:5]}")  # Log the first 5 ITE values
    else:
        nuisance["tarnet"] = None

    if "af_wstein" in config_af:
        # print("Train Repr net wasserstein")
        utils.set_seed(seed)
        config_repr_wstein = get_hyper(config_exp, datasets["d_train"], hyper_path + "/nuisance/repr_net_wstein")
        if fixed_params is not None:
            if "gamma" in fixed_params:
                config_repr_wstein["model"]["gamma"] = fixed_params["gamma"]
        repr_wstein = fp.train_fair_repr(datasets, config_repr_wstein, loss="wstein")
        nuisance["repr_wstein"] = repr_wstein["trained_model"]
        # print(f"Repr Wasserstein trained: {repr_wstein}")  # Log training result

    if "af_conf" in config_af:
        # print("Train Repr net domain confusion")
        utils.set_seed(seed)
        config_repr_conf = get_hyper(config_exp, datasets["d_train"], hyper_path + "/nuisance/repr_net_conf")
        if fixed_params is not None:
            if "gamma" in fixed_params:
                config_repr_conf["model"]["gamma"] = fixed_params["gamma"]
        repr_conf = fp.train_fair_repr(datasets, config_repr_conf, loss="conf")
        nuisance["repr_conf"] = repr_conf["trained_model"]
        # print(f"Repr Domain Confusion trained: {repr_conf}")  # Log training result

    if "af_gr" in config_af:
        # print("Train Repr net gradient reversal")
        utils.set_seed(seed)
        config_repr_gr = get_hyper(config_exp, datasets["d_train"], hyper_path + "/nuisance/repr_net_gr")
        if fixed_params is not None:
            if "gamma" in fixed_params:
                config_repr_gr["model"]["gamma"] = fixed_params["gamma"]
        repr_gr = fp.train_fair_repr(datasets, config_repr_gr, loss="gr")
        nuisance["repr_gr"] = repr_gr["trained_model"]
        # print(f"Repr Gradient Reversal trained: {repr_gr}")  # Log training result

    # Train policy nets
    # print("Policy net training--------------------------------")
    trained_models = {}
    for model_config in model_configs:
        utils.set_seed(seed)

        if model_config["name"] == "untrained":
            config_unfair = get_hyper(config_exp, datasets["d_train"], hyper_path + "/policy_nets/fpnet_untrained")
            policy_unfair = ub.PoliyNetUnfair(config_unfair)
            trained_models["untrained"] = {"trained_model": policy_unfair}

        if model_config["name"] == "oracle":
            oracle_trained = ub.OraclePolicy()
            trained_models["oracle"] = {"trained_model": oracle_trained}

        if model_config["name"] == "oracle_af":
            oracle_trained = ub.OraclePolicy_af()
            trained_models["oracle_af"] = {"trained_model": oracle_trained}

        if model_config["name"] == "oracle_trained":
            # print("Trained oracle")
            config_oracle = get_hyper(config_exp, datasets["d_train"], hyper_path + "/policy_nets/oracle")
            oracle_trained = ub.train_policynet_unfair(datasets, config_oracle)
            trained_models["oracle_trained"] = oracle_trained

        if model_config["name"] == "fpnet":
            af = model_config["action_fair"]
            vf = model_config["value_fair"]
            m = config_exp["experiment"]["m"]
            model_name = "fpnet_" + vf + "_" + m
            # print(f"Train {model_name}")
            config_fpnet = get_hyper(config_exp, datasets["d_train"],
                                            hyper_path + "/policy_nets/" + af + "/" + model_name)
            model_name = model_name + "_" + af
            config_fpnet["model"]["af"] = af
            config_fpnet["model"]["vf"] = vf
            config_fpnet["model"]["m"] = m
            if fixed_params is not None:
                if "lamb" in fixed_params:
                    config_fpnet["model"]["lamb"] = fixed_params["lamb"]
                if "delta" in fixed_params:
                    config_fpnet["model"]["delta"] = fixed_params["delta"]
            utils.set_seed(seed)
            if af == "af_wstein":
                trained_models[model_name] = fp.train_fpnet(datasets, config_fpnet, nuisance["tarnet"], nuisance["repr_wstein"])
            elif af == "af_conf":
                trained_models[model_name] = fp.train_fpnet(datasets, config_fpnet, nuisance["tarnet"], nuisance["repr_conf"])
            elif af == "af_gr":
                trained_models[model_name] = fp.train_fpnet(datasets, config_fpnet, nuisance["tarnet"], nuisance["repr_gr"])
            else:
                trained_models[model_name] = fp.train_fpnet(datasets, config_fpnet, nuisance["tarnet"])

        if model_config["name"] == "ols":
            model_name = "ols"
            ols_model = OLSModel(standardize=True, interaction_covariates=config_exp["data"]["covariate_cols"])

            # Extrahiere Features und Zielvariable
            X_train_tensor = datasets["d_train"].data["x"]
            a_train = datasets["d_train"].data["a"].cpu().numpy().ravel()
            y_train_tensor = datasets["d_train"].data["y"]

            # Hole die echten Feature-Namen aus der Config!
            feature_names = config_exp["data"]["covariate_cols"] + ["assignment"]

            # Erstelle DataFrame mit echten Namen
            X_train_df = pd.DataFrame(X_train_tensor.cpu().numpy(), columns=config_exp["data"]["covariate_cols"])
            X_train_df["assignment"] = a_train
            X_train_df = X_train_df[feature_names]  # richtige Reihenfolge

            y_train_series = pd.Series(y_train_tensor.cpu().numpy().ravel())

            # Trainiere das OLS-Modell
            ols_model.train(X_train_df, y_train_series)
            trained_models[model_name] = {"trained_model": ols_model}

    return trained_models


def get_hyper(config_exp, d_train, path_hyper):
    config_data = {"data": {}}
    config_data["data"]["x_dim"] = d_train.data["x"].shape[1]
    config_data["data"]["s_dim"] = d_train.data["s"].shape[1]
    config_data["data"]["y_type"] = d_train.datatypes["y_type"]
    config_data["data"]["xu_dim"] = d_train.dim_xu
    config_model = {}
    config_model["model"] = utils.load_yaml(path_hyper)
    config_hyper = config_exp | config_model | config_data
    # print(f"Loaded hyperparameters: {config_hyper}")  # Log hyperparameters for debugging
    return config_hyper
