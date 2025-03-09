import utils
from hyperparam.main import perform_hyper_tuning

if __name__ == "__main__":
    config_hyper_real = utils.load_yaml("/hyperparam/exp_real/config_hyper_real")
    perform_hyper_tuning(config_hyper_real)