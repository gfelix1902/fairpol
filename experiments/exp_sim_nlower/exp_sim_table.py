from experiments.experiment import perform_experiment
import utils
import joblib

if __name__ == "__main__":
    config_exp = utils.load_yaml("/experiments/exp_sim_nlower/config_sim")
    results = perform_experiment(config_exp)
    if config_exp["experiment"]["save_results"]:
        # Save results to file
        m = config_exp["experiment"]["m"]
        path_results = utils.get_project_path() + "/results/exp_sim_nlower/table/"
        joblib.dump(results["pvalues"], path_results + "pvalues_" + m + ".pkl")
        joblib.dump(results["pvalues0"], path_results + "pvalues0_" + m + ".pkl")
        joblib.dump(results["pvalues1"], path_results + "pvalues1_" + m + ".pkl")
        joblib.dump(results["af"], path_results + "af_" + m + ".pkl")