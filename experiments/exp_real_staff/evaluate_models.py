from experiments.model_evaluation import get_policy_predictions, get_table_pvalues_conditional
import utils
import joblib
import os

if __name__ == "__main__":
    try:
        config_exp = utils.load_yaml("/experiments/exp_real_staff/config_real_staff")
    except FileNotFoundError:
        print("❌ Fehler: Konfigurationsdatei nicht gefunden.")
        exit()

    config_exp["data"]["dataset"] = "real_staff"

    try:
        # Lade die Daten
        datasets = utils.load_data(config_exp["data"], seed=config_exp["experiment"]["seed"])

        # Lade die trainierten Modelle
        path_models = utils.get_project_path() + "/results/exp_real_staff/models/"
        trained_models = joblib.load(os.path.join(path_models, "trained_models.pkl"))
        print(f"✅ Trainierte Modelle geladen aus: {path_models}")

        # Führe die Evaluation durch
        policy_predictions = get_policy_predictions(trained_models, datasets["d_test"], data_type=config_exp["data"]["dataset"])
        pvalues_conditional = get_table_pvalues_conditional(trained_models, datasets["d_test"], data_type=config_exp["data"]["dataset"])

        # Speicherort für die Ergebnisse
        path_results = utils.get_project_path() + "/results/exp_real_staff/table/"
        os.makedirs(path_results, exist_ok=True)

        # Speichere die Ergebnisse
        joblib.dump(policy_predictions, os.path.join(path_results, "policy_predictions.pkl"))
        joblib.dump(pvalues_conditional, os.path.join(path_results, "pvalues_conditional.pkl"))
        print(f"💾 Ergebnisse gespeichert unter: {path_results}")

    except Exception as e:
        print(f"❌ Fehler während der Evaluation: {e}")