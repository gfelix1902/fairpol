import utils
from hyperparam.main import perform_hyper_tuning

if __name__ == "__main__":
    config_hyper_test = utils.load_yaml("/hyperparam/exp_sim/config_hyper_sim")
    perform_hyper_tuning(config_hyper_test)