import utils
from optuna.samplers import RandomSampler
import optuna
import models.fp_net as fp
import models.unfair_baselines as ub
import random
import torch
import numpy as np


def perform_hyper_tuning(config):
    # Load data
    config_data = config["data"]
    datasets = utils.load_data(config_data)
    #Experiment configuration
    config_exp = config.copy()
    config_exp.pop("data")
    config_exp.pop("tuning_ranges")
    #Other stuff
    num_runs = config["experiment"]["runs"]
    model_configs = config["experiment"]["models"]
    tuning_ranges = config["tuning_ranges"]
    hyper_path = config["experiment"]["hyper_path"]

    # nuisance models--------------------------
    nuisance = {}
    #TARNet
    if config["experiment"]["tune_tarnet"]:
        params_tarnet = tune_tarnet(datasets, tuning_ranges["tarnet"], config_exp, num_samples=num_runs,
                               hyper_path=hyper_path, seed=config["experiment"]["seed"])
    else:
        params_tarnet = params_to_config(params=utils.load_yaml(hyper_path + "/nuisance/tarnet"), config_exp=config_exp,
                                  d_train=datasets["d_train"])
    if not config["experiment"]["nuisance_only"]:
        nuisance["tarnet"] = ub.train_tarnet(datasets, params_tarnet)["trained_model"]

    # Check need for tuning representation learning
    config_af = utils.get_config_af(model_configs)
    if "af_conf" in config_af:
        if config["experiment"]["tune_repr"]:
            params = tune_repr_net(datasets, tuning_ranges["repr_net_conf"], config_exp, loss="conf", num_samples=num_runs,
                                   hyper_path=hyper_path,
                                   seed=config["experiment"]["seed"])
        else:
            params = params_to_config(params=utils.load_yaml(hyper_path + "/nuisance/repr_net_conf"),config_exp=config_exp,
                                      d_train=datasets["d_train"])
        if not config["experiment"]["nuisance_only"]:
            nuisance["repr_net_conf"] = fp.train_fair_repr(datasets, params, loss="conf")["trained_model"]
    if "af_wstein" in config_af:
        if config["experiment"]["tune_repr"]:
            params = tune_repr_net(datasets, tuning_ranges["repr_net_wstein"], config_exp, loss="wstein", num_samples=num_runs,
                                   hyper_path=hyper_path,
                                   seed=config["experiment"]["seed"])
        else:
            params = params_to_config(params=utils.load_yaml(hyper_path + "/nuisance/repr_net_wstein"),
                                      config_exp=config_exp,
                                      d_train=datasets["d_train"])
        if not config["experiment"]["nuisance_only"]:
            nuisance["repr_net_wstein"] = fp.train_fair_repr(datasets, params, loss="wstein")["trained_model"]
    if "af_gr" in config_af:
        if config["experiment"]["tune_repr"]:
            params = tune_repr_net(datasets, tuning_ranges["repr_net_gr"], config_exp, loss="gr", num_samples=num_runs,
                                   hyper_path=hyper_path,
                                   seed=config["experiment"]["seed"])
        else:
            params = params_to_config(params=utils.load_yaml(hyper_path + "/nuisance/repr_net_gr"),
                                      config_exp=config_exp,
                                      d_train=datasets["d_train"])
        if not config["experiment"]["nuisance_only"]:
            nuisance["repr_net_gr"] = fp.train_fair_repr(datasets, params, loss="gr")["trained_model"]

    if not config["experiment"]["nuisance_only"]:
        for model_config in config["experiment"]["models"]:
            tuning_range_name = "fpnet_" + model_config["value_fair"]
            tune_policy_net(datasets, tuning_ranges[tuning_range_name], config_exp, model_config, nuisance, num_samples=num_runs,
                            hyper_path=hyper_path, seed=config["experiment"]["seed"])


def tune_policy_net(datasets, tuning_ranges, config_exp, model_config, nuisance, num_samples=10, hyper_path="/hyperparam/exp_sim",
                    seed=0):
    repr_net = None
    if model_config["action_fair"] == "af_conf":
        repr_net = nuisance["repr_net_conf"]
    if model_config["action_fair"] == "af_wstein":
        repr_net = nuisance["repr_net_wstein"]
    if model_config["action_fair"] == "af_gr":
        repr_net = nuisance["repr_net_gr"]
    tune_sampler = set_seeds(seed)
    obj_repr_net = get_objective_policy_net(datasets, tuning_ranges, model_config, config_exp, repr_net=repr_net, tarnet=nuisance["tarnet"])
    study_name = "fpnet_" + model_config["value_fair"] + "_" + model_config["m"]
    study = tune_objective(obj_repr_net, study_name, num_samples, tune_sampler)
    best_params = study.best_trial.params
    #Save params
    path = hyper_path + "/policy_nets/" + model_config["action_fair"] + "/" + study_name
    utils.save_yaml(path, best_params)
    return params_to_config(best_params, config_exp, datasets["d_train"])


def tune_repr_net(datasets, tuning_ranges, config_exp, loss="conf", num_samples=10, hyper_path="/hyperparam/exp_sim", seed=0):
    tune_sampler = set_seeds(seed)
    obj_repr_net = get_objective_repr_net(datasets, tuning_ranges, config_exp, loss=loss)
    study_name = "repr_net_" + loss
    study = tune_objective(obj_repr_net, study_name, num_samples, tune_sampler)
    best_params = study.best_trial.params
    #Save params
    path = hyper_path + "/nuisance/repr_net_" + loss
    utils.save_yaml(path, best_params)
    return params_to_config(best_params, config_exp, datasets["d_train"])


def tune_tarnet(datasets, tuning_ranges, config_exp, num_samples=10, hyper_path="/hyperparam/exp_sim", seed=0):
    tune_sampler = set_seeds(seed)
    obj_repr_net = get_objective_tarnet(datasets, tuning_ranges, config_exp)
    study_name = "tarnet"
    study = tune_objective(obj_repr_net, study_name, num_samples, tune_sampler)
    best_params = study.best_trial.params
    #Save params
    path = hyper_path + "/nuisance/tarnet"
    utils.save_yaml(path, best_params)
    return params_to_config(best_params, config_exp, datasets["d_train"])


def tune_objective(objective, study_name, num_samples=10, sampler=None):
    if sampler is not None:
        study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler)
    else:
        study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(objective, n_trials=num_samples)

    print("Finished. Best trial:")
    trial_best = study.best_trial

    print("  Value: ", trial_best.value)

    print("  Params: ")
    for key, value in trial_best.params.items():
        print("    {}: {}".format(key, value))
    # save_dir = path + study_name + ".pkl"
    # joblib.dump(study, save_dir)
    return study


def get_objective_policy_net(datasets, tuning_ranges, model_config, config_exp, repr_net=None, tarnet=None):
    def obj(trial):
        config = sample_config(trial, tuning_ranges, config_exp, datasets["d_train"])
        config["model"]["af"] = model_config["action_fair"]
        config["model"]["vf"] = model_config["value_fair"]
        config["model"]["m"] = model_config["m"]
        model = fp.train_fpnet(datasets, config, tarnet=tarnet, repr_net=repr_net)
        return model["val_results"]["val_obj"]

    return obj


def get_objective_repr_net(datasets, tuning_ranges, config_exp, loss="conf"):
    def obj(trial):
        config = sample_config(trial, tuning_ranges, config_exp, datasets["d_train"])
        model = fp.train_fair_repr(datasets, config, loss=loss)
        return model["val_results"]["val_obj"]
    return obj

def get_objective_tarnet(datasets, tuning_ranges, config_exp):
    def obj(trial):
        config = sample_config(trial, tuning_ranges, config_exp, datasets["d_train"])
        model = ub.train_tarnet(datasets, config)
        return model["val_results"]["val_obj"]
    return obj


def sample_config(trial, tuning_ranges, config_exp, d_train):
    params = {}
    for param in tuning_ranges.keys():
        params[param] = trial.suggest_categorical(param, tuning_ranges[param])
    return params_to_config(params, config_exp, d_train)

def params_to_config(params, config_exp, d_train):
    config_data = {"data": {}}
    config_data["data"]["x_dim"] = d_train.data["x"].shape[1]
    config_data["data"]["s_dim"] = d_train.data["s"].shape[1]
    config_data["data"]["y_type"] = d_train.datatypes["y_type"]

    config_data["data"]["xu_dim"] = d_train.dim_xu
    config_hyper = {}
    config_hyper["model"] = params
    config_hyper["experiment"] = config_exp["experiment"]
    config_hyper["data"] = config_data["data"]
    return config_hyper


def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tune_sampler = RandomSampler(seed=seed)
    return tune_sampler
