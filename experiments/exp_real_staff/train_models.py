from experiments.model_training import train_models
import utils
import joblib
import os
import torch

def move_to_cpu(obj):
    if hasattr(obj, "to"):
        try:
            obj = obj.to("cpu")
        except Exception:
            pass
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_cpu(v) for v in obj)
    return obj

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

        # Trainiere die Modelle
        trained_models = train_models(config_exp, datasets, seed=config_exp["experiment"]["seed"])

        # Modelle vor dem Speichern auf CPU verschieben
        trained_models_cpu = move_to_cpu(trained_models)

        # Speicherort für die trainierten Modelle
        path_models = utils.get_project_path() + "/results/exp_real_staff/models/"
        os.makedirs(path_models, exist_ok=True)

        # Speichere die trainierten Modelle
        joblib.dump(trained_models_cpu, os.path.join(path_models, "trained_models.pkl"))
        print(f"💾 Trainierte Modelle gespeichert unter: {path_models}")

    except Exception as e:
        print(f"❌ Fehler während des Trainings: {e}")
