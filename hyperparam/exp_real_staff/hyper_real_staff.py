import utils
from hyperparam.main import perform_hyper_tuning

if __name__ == "__main__":
    config_hyper_real_staff = utils.load_yaml("/hyperparam/exp_real_staff/config_hyper_real_staff") # Pfad anpassen
    perform_hyper_tuning(config_hyper_real_staff)