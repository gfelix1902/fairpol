import utils
import models.fp_net as fp
import models.unfair_baselines as ub


def train_models(config_exp, datasets, seed=1, fixed_params=None):
    hyper_path = config_exp["experiment"]["hyper_path"]
    model_configs = config_exp["experiment"]["models"]
    config_af = utils.get_config_af(model_configs)
    nuisance = {}

    # Train nuisance models
    print("Nuisance training----------------------------------")

    if not config_exp["experiment"]["oracle_nuisance"]:
        # Nuisance model TARNet
        print("Train nuisance TARNet")
        utils.set_seed(seed)
        config_tarnet = get_hyper(config_exp, datasets["d_train"], hyper_path + "/nuisance/tarnet")
        tarnet = ub.train_tarnet(datasets, config_tarnet)
        nuisance["tarnet"] = tarnet["trained_model"]
        tarnet_pred = nuisance["tarnet"].predict_nuisance(datasets["d_test"].data)
        tarnet_predm1 = tarnet_pred["mu1"].detach().numpy()
        tarnet_predm0 = tarnet_pred["mu0"].detach().numpy()
        tarnet_predp = tarnet_pred["prop"].detach().numpy()
        pred_ite = tarnet_predm1 - tarnet_predm0
    else:
        nuisance["tarnet"] = None

    if "af_wstein" in config_af:
        print("Train Repr net wasserstein")
        utils.set_seed(seed)
        config_repr_wstein = get_hyper(config_exp, datasets["d_train"], hyper_path + "/nuisance/repr_net_wstein")
        if fixed_params is not None:
            if "gamma" in fixed_params:
                config_repr_wstein["model"]["gamma"] = fixed_params["gamma"]
        repr_wstein = fp.train_fair_repr(datasets, config_repr_wstein, loss="wstein")
        nuisance["repr_wstein"] = repr_wstein["trained_model"]

    if "af_conf" in config_af:
        print("Train Repr net domain confusion")
        utils.set_seed(seed)
        config_repr_conf = get_hyper(config_exp, datasets["d_train"], hyper_path + "/nuisance/repr_net_conf")
        if fixed_params is not None:
            if "gamma" in fixed_params:
                config_repr_conf["model"]["gamma"] = fixed_params["gamma"]
        repr_wstein = fp.train_fair_repr(datasets, config_repr_conf, loss="conf")
        nuisance["repr_conf"] = repr_wstein["trained_model"]

    if "af_gr" in config_af:
        print("Train Repr net gradient reversal")
        utils.set_seed(seed)
        config_repr_gr = get_hyper(config_exp, datasets["d_train"], hyper_path + "/nuisance/repr_net_gr")
        if fixed_params is not None:
            if "gamma" in fixed_params:
                config_repr_gr["model"]["gamma"] = fixed_params["gamma"]
        repr_wstein = fp.train_fair_repr(datasets, config_repr_gr, loss="gr")
        nuisance["repr_gr"] = repr_wstein["trained_model"]

    # Train policy nets
    print("Policy net training--------------------------------")
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
            print("Trained oracle")
            config_oracle = get_hyper(config_exp, datasets["d_train"], hyper_path + "/policy_nets/oracle")
            oracle_trained = ub.train_policynet_unfair(datasets, config_oracle)
            trained_models["oracle_trained"] = oracle_trained

        if model_config["name"] == "fpnet":
            af = model_config["action_fair"]
            vf = model_config["value_fair"]
            m = config_exp["experiment"]["m"]
            model_name = "fpnet_" + vf + "_" + m
            print(f"Train {model_name}")
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
    return config_hyper
