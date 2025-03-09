from experiments import experiment
import utils
import joblib

if __name__ == "__main__":
    delta_range = [0]
    try:
        config_exp = utils.load_yaml("/experiments/exp_real_staff/config_real_staff")
    except FileNotFoundError:
        print("Fehler: Konfigurationsdatei nicht gefunden.")
        exit()

    config_exp["data"]["dataset"] = "real_staff"

    for delta in delta_range:
        results = experiment.perform_experiment(config_exp, fixed_params={"delta": delta})
        if config_exp["experiment"]["save_results"]:
            path_results = utils.get_project_path() + "/results/exp_real_staff/table/"
            joblib.dump(results["pvalues"], path_results + "pvalues.pkl")
            joblib.dump(results["pvalues0"], path_results + "pvalues0.pkl")
            joblib.dump(results["pvalues1"], path_results + "pvalues1.pkl")
            joblib.dump(results["af"], path_results + "af.pkl")
            joblib.dump(results["predictions"], path_results + "predictions.pkl")