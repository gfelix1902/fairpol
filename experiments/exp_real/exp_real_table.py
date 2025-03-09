from experiments import experiment
import utils
import joblib

if __name__ == "__main__":
    #Budget penalty
    delta_range = [0]
    config_exp = utils.load_yaml("/experiments/exp_real/config_real")
    for delta in delta_range:
        results = experiment.perform_experiment(config_exp, fixed_params={"delta": delta})
        if config_exp["experiment"]["save_results"]:
            # Save results to file
            path_results = utils.get_project_path() + "/results/exp_real/table/"
            joblib.dump(results["pvalues"], path_results + "pvalues.pkl")
            joblib.dump(results["pvalues0"], path_results + "pvalues0.pkl")
            joblib.dump(results["pvalues1"], path_results + "pvalues1.pkl")
            joblib.dump(results["af"], path_results + "af.pkl")
            joblib.dump(results["predictions"], path_results + "predictions.pkl")