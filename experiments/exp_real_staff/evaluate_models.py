from experiments.model_evaluation import get_policy_predictions, get_table_pvalues_conditional
import utils
import joblib
import os
import pandas as pd

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
        
        # Speicherort für die Ergebnisse
        path_results = utils.get_project_path() + "/results/exp_real_staff/table/"
        os.makedirs(path_results, exist_ok=True)

        # Führe die Evaluation durch
        policy_predictions = get_policy_predictions(trained_models, datasets["d_test"], data_type=config_exp["data"]["dataset"])
        pvalues_conditional = get_table_pvalues_conditional(trained_models, datasets["d_test"], data_type=config_exp["data"]["dataset"])

        # Zusätzliche Ausgabe: Für jedes Modell, jedes Testdatenelement
        print("\n--- Detailausgabe pro Modell und Testdatenelement ---")
        y = datasets["d_test"].data["y"]      # Outcome
        a = datasets["d_test"].data["a"]      # Treatment
        n = len(y)
        for model_name in policy_predictions:
            print(f"\nModell: {model_name}")
            preds = policy_predictions[model_name]
            model = trained_models[model_name]
            if isinstance(model, dict) and "trained_model" in model:
                model = model["trained_model"]

            # OLSModel benötigt ein DataFrame als Input für predict_ite
            if model_name == "ols":
                # Extrahiere das DataFrame aus deinem Static_Dataset
                X_test = pd.DataFrame(datasets["d_test"].data["x"].cpu().numpy())
                # Füge die Treatment-Spalte hinzu
                X_test["assignment"] = datasets["d_test"].data["a"].cpu().numpy().ravel()
                # Setze alle Spaltennamen auf String
                X_test.columns = X_test.columns.astype(str)
                ite = model.predict_ite(X_test, treat_col="assignment")
            else:
                if not hasattr(model, "predict_ite"):
                    print(f"Modell {model_name} unterstützt keine ITE-Schätzung.")
                    continue
                ite = model.predict_ite(datasets["d_test"])

            # Ergebnisse sammeln
            results = []
            for idx in range(n):
                treatment_val = a[idx].item() if hasattr(a[idx], "item") else a[idx]
                outcome_val = y[idx].item() if hasattr(y[idx], "item") else y[idx]
                policy_value = preds[idx] if idx < len(preds) else "N/A"
                ite_val = ite[idx].item() if hasattr(ite[idx], "item") else ite[idx]
                results.append({
                    "Index": idx,
                    "Treatment": treatment_val,
                    "Outcome": outcome_val,
                    "Policy_Vorhersage": policy_value,
                    "ITE": ite_val
                })
            # Speichere als CSV
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(path_results, f"detailausgabe_{model_name}.csv"), index=False)
            print(f"💾 Detailausgabe für {model_name} gespeichert unter: {os.path.join(path_results, f'detailausgabe_{model_name}.csv')}")


        # Speichere die Ergebnisse
        joblib.dump(policy_predictions, os.path.join(path_results, "policy_predictions.pkl"))
        joblib.dump(pvalues_conditional, os.path.join(path_results, "pvalues_conditional.pkl"))
        print(f"💾 Ergebnisse gespeichert unter: {path_results}")

    except Exception as e:
        print(f"❌ Fehler während der Evaluation: {e}")