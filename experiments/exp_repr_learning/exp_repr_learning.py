from experiments.experiment import perform_experiment
import utils
import joblib

if __name__ == "__main__":
    config_exp = utils.load_yaml("/experiments/exp_repr_learning/config_repr_learning")
    i = 0
    gamma_gr = [0.2, 0.3, 0.4, 0.5, 0.6]
    gamma_conf = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
    gamma_wstein = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
    results = perform_experiment(config_exp, {"gamma_gr": gamma_gr[i], "gamma_conf": gamma_conf[i], "gamma_wstein": gamma_wstein[i]})
    if config_exp["experiment"]["save_results"]:
        # Save results to file
        m = config_exp["experiment"]["m"]
        path_results = utils.get_project_path() + "/results/exp_repr_learning/table/"
        joblib.dump(results["pvalues"], path_results + "pvalues_" + m + f'_1_{i}' + ".pkl")
        joblib.dump(results["pvalues0"], path_results + "pvalues0_" + m + f'_1_{i}' + ".pkl")
        joblib.dump(results["pvalues1"], path_results + "pvalues1_" + m + f'_1_{i}' + ".pkl")
        joblib.dump(results["af"], path_results + "af_" + m + f'_1_{i}' + ".pkl")
