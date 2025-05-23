from experiments.model_evaluation import get_policy_predictions, get_table_pvalues_conditional
import utils
import joblib
import os
import pandas as pd
import numpy as np
import torch

def move_model_to_cpu(model):
    # Falls das Modell ein dict mit "trained_model" ist
    if isinstance(model, dict) and "trained_model" in model:
        model_obj = model["trained_model"]
        if hasattr(model_obj, "model") and hasattr(model_obj.model, "to"):
            model_obj.model.to(torch.device('cpu'))
    # Falls das Modell direkt ein torch.nn.Module ist
    elif hasattr(model, "to"):
        model.to(torch.device('cpu'))

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
        trained_models = joblib.load(os.path.join(path_models, "trained_models.pkl"), mmap_mode=None)
        for m in trained_models.values():
            move_model_to_cpu(m)
        print(f"✅ Trainierte Modelle geladen aus: {path_models}")
        
        # Speicherort für die Ergebnisse
        path_results = utils.get_project_path() + "/results/exp_real_staff/table/"
        os.makedirs(path_results, exist_ok=True)

        # Führe die Evaluation durch
        policy_predictions = get_policy_predictions(trained_models, datasets["d_test"], data_type=config_exp["data"]["dataset"])
        pvalues_conditional = get_table_pvalues_conditional(trained_models, datasets["d_test"], data_type=config_exp["data"]["dataset"])

        print("\n--- Detailausgabe pro Modell und Testdatenelement ---")
        y = datasets["d_test"].data["y"]
        a = datasets["d_test"].data["a"]
        n = len(y)

        for model_name, preds in policy_predictions.items():
            print(f"\nModell: {model_name}")
            model = trained_models[model_name]
            if isinstance(model, dict) and "trained_model" in model:
                model = model["trained_model"]

            results = []

            # Feature-Namen bestimmen (für alle Modelle, nicht nur OLS)
            if hasattr(model, "feature_order") and model.feature_order is not None:
                feature_names = model.feature_order
            elif "covariate_cols" in config_exp["data"]:
                feature_names = config_exp["data"]["covariate_cols"]
            else:
                raise ValueError("Feature-Namen konnten nicht bestimmt werden!")

            # OLS-Modell: DataFrame für predict_ite und predict_cate vorbereiten
            if model_name == "ols":
                feature_names = config_exp["data"]["covariate_cols"] + ["assignment"]
                # Erstelle DataFrame nur mit den Kovariaten
                X_test = pd.DataFrame(
                    datasets["d_test"].data["x"].cpu().numpy(),
                    columns=config_exp["data"]["covariate_cols"]
                )
                # Füge assignment als neue Spalte hinzu
                X_test["assignment"] = datasets["d_test"].data["a"].cpu().numpy().ravel()
                
                # WICHTIG: Interaktionsterm hinzufügen, wenn er im Training verwendet wurde!
                if "trainy1" in X_test.columns and "trainy2" in X_test.columns:
                    X_test["trainy1_x_trainy2"] = X_test["trainy1"] * X_test["trainy2"]
                
                # Prüfe, ob alle Trainings-Features vorhanden sind
                if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
                    for feature in model.feature_names_in_:
                        if feature not in X_test.columns:
                            print(f"⚠️ Warnung: Feature '{feature}' fehlt im Test-DataFrame! Wird mit Nullen ergänzt.")
                            X_test[feature] = 0
                
                # Sortiere die Spalten in die richtige Reihenfolge
                if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
                    # Verwende exakt die gleichen Features wie beim Training
                    X_test = X_test[model.feature_names_in_]
                else:
                    # Fallback zur Standard-Reihenfolge
                    X_test = X_test[feature_names]
                    
                ite = model.predict_ite(X_test, treat_col="assignment")
                cate_both = model.predict_cate(X_test, treat_cols=["trainy1", "trainy2"], treat_values=[1, 1], base_values=[0, 0])
                cate_first = model.predict_cate(X_test, treat_cols=["trainy1", "trainy2"], treat_values=[1, 0], base_values=[0, 0])
                cate_second = model.predict_cate(X_test, treat_cols=["trainy1", "trainy2"], treat_values=[0, 1], base_values=[0, 0])
                print("CATE (beide Jahre vs. kein Training):", np.mean(cate_both))
                print("CATE (nur 1. Jahr):", np.mean(cate_first))
                print("CATE (nur 2. Jahr):", np.mean(cate_second))
            # Andere Modelle mit ITE/CATE
            elif hasattr(model, "predict_ite") and hasattr(model, "predict_cate"):
                ite = model.predict_ite(datasets["d_test"])
                cate_both = model.predict_cate(
                    datasets["d_test"],
                    treat_cols=["trainy1", "trainy2"],
                    treat_values=[1, 1],
                    base_values=[0, 0],
                    feature_names=feature_names
                )
                cate_first = model.predict_cate(
                    datasets["d_test"],
                    treat_cols=["trainy1", "trainy2"],
                    treat_values=[1, 0],
                    base_values=[0, 0],
                    feature_names=feature_names
                )
                cate_second = model.predict_cate(
                    datasets["d_test"],
                    treat_cols=["trainy1", "trainy2"],
                    treat_values=[0, 1],
                    base_values=[0, 0],
                    feature_names=feature_names
                )
                print("CATE (beide Jahre vs. kein Training):", np.mean(cate_both))
                print("CATE (nur 1. Jahr):", np.mean(cate_first))
                print("CATE (nur 2. Jahr):", np.mean(cate_second))
            else:
                print(f"Modell {model_name} unterstützt keine ITE/CATE-Schätzung.")
                continue

            # Ergebnisse sammeln und speichern
            for idx in range(n):
                treatment_val = a[idx].item() if hasattr(a[idx], "item") else a[idx]
                outcome_val = y[idx].item() if hasattr(y[idx], "item") else y[idx]
                policy_value = preds[idx] if idx < len(preds) else "N/A"
                ite_val = ite[idx].item() if hasattr(ite[idx], "item") else ite[idx]
                cate_both_val = cate_both[idx] if len(cate_both) > idx else np.nan
                cate_first_val = cate_first[idx] if len(cate_first) > idx else np.nan
                cate_second_val = cate_second[idx] if len(cate_second) > idx else np.nan
                results.append({
                    "Index": idx,
                    "Treatment": treatment_val,
                    "Outcome": outcome_val,
                    "Policy_Vorhersage": policy_value,
                    "ITE": ite_val,
                    "CATE_both": cate_both_val,
                    "CATE_first": cate_first_val,
                    "CATE_second": cate_second_val
                })
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(path_results, f"detailausgabe_{model_name}.csv"), index=False)
            print(f"💾 Detailausgabe für {model_name} gespeichert unter: {os.path.join(path_results, f'detailausgabe_{model_name}.csv')}")

        # Speichere die Ergebnisse
        joblib.dump(policy_predictions, os.path.join(path_results, "policy_predictions.pkl"))
        joblib.dump(pvalues_conditional, os.path.join(path_results, "pvalues_conditional.pkl"))
        print(f"💾 Ergebnisse gespeichert unter: {path_results}")

    except Exception as e:
        print(f"❌ Fehler während der Evaluation: {e}")