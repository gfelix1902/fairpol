from experiments import experiment
import utils
import joblib
import os

if __name__ == "__main__":
    delta_range = [0]
    
    try:
        config_exp = utils.load_yaml("/experiments/exp_real_staff/config_real_staff")
    except FileNotFoundError:
        print("❌ Fehler: Konfigurationsdatei nicht gefunden.")
        exit()

    config_exp["data"]["dataset"] = "real_staff"

    for delta in delta_range:
        print(f"🚀 Starte Experiment mit delta = {delta}...")

        # Debug-Ausgabe der Konfiguration
        print("🔎 Konfiguration:", config_exp)

        # Führe das Experiment aus
        try:
            results = experiment.perform_experiment(config_exp, fixed_params={"delta": delta})
            if results is None:
                print("⚠️ Keine Ergebnisse zurückgegeben!")
                continue

            # Debug-Ausgabe der Ergebnisse
            print(f"✅ Experiment abgeschlossen. Ergebnisse: {results.keys()}")

            # Sicherheitscheck: Stelle sicher, dass alle Schlüssel vorhanden sind
            required_keys = ["pvalues", "pvalues0", "pvalues1", "af", "predictions"]
            missing_keys = [key for key in required_keys if key not in results]

            if missing_keys:
                print(f"⚠️ Fehlende Schlüssel in den Ergebnissen: {missing_keys}")
                continue

            # Speicherort erstellen, wenn nicht vorhanden
            path_results = utils.get_project_path() + "/results/exp_real_staff/table/"
            os.makedirs(path_results, exist_ok=True)

            # Ergebnisse speichern
            joblib.dump(results["pvalues"], os.path.join(path_results, "pvalues.pkl"))
            joblib.dump(results["pvalues0"], os.path.join(path_results, "pvalues0.pkl"))
            joblib.dump(results["pvalues1"], os.path.join(path_results, "pvalues1.pkl"))
            joblib.dump(results["af"], os.path.join(path_results, "af.pkl"))
            joblib.dump(results["predictions"], os.path.join(path_results, "predictions.pkl"))

            print(f"💾 Ergebnisse gespeichert unter: {path_results}")

        except Exception as e:
            print(f"❌ Fehler während der Experimentdurchführung: {e}")