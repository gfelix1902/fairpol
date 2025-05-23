import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def get_models_names(trained_models):
    models_names = []
    for model in trained_models.keys():
        models_names.append({"name": model, "model": trained_models[model]["trained_model"]})
    return models_names


def create_table(models_names, rows):
    names = [model["name"] for model in models_names]
    df_results = pd.DataFrame(columns=names, index=range(rows))
    return df_results


def get_policy_predictions(trained_models, d_test, data_type="real", covariate_cols_from_config=None): # MODIFIED SIGNATURE
    models_names = get_models_names(trained_models)
    df_results = pd.DataFrame(columns=[model["name"] for model in models_names], index=range(d_test.data["y"].shape[0]))

    for model in models_names:
        model_name = model["name"]
        model_instance = model["model"]

        if model_name == "ols":
            X_test_tensor = d_test.data["x"] # This contains only the covariate values

            if covariate_cols_from_config is None:
                raise ValueError("covariate_cols_from_config must be provided for OLS model evaluation when creating DataFrame from X_test_tensor.")
            
            # Create DataFrame with the correct covariate names for X_test_tensor
            # X_test_tensor has shape (n_samples, n_covariates)
            X_test_df = pd.DataFrame(X_test_tensor.cpu().numpy(), columns=covariate_cols_from_config)
            
            # Add 'assignment' column from d_test.data["a"]
            X_test_df["assignment"] = d_test.data["a"].cpu().numpy().ravel()
            
            # Add interaction term if its components ("trainy1", "trainy2") are present
            # This check ensures that we only attempt to create the interaction if the base columns exist
            if "trainy1" in X_test_df.columns and "trainy2" in X_test_df.columns:
                X_test_df["trainy1_x_trainy2"] = X_test_df["trainy1"] * X_test_df["trainy2"]

            # Align DataFrame with the features expected by the trained OLS model
            train_columns = getattr(model_instance, "feature_names_in_", None)
            if train_columns is not None:
                train_columns = list(train_columns) # Ensure it's a list
                # Add any missing columns that the model expects (e.g., if interaction wasn't created above but model expects it)
                for col in train_columns:
                    if col not in X_test_df.columns:
                        print(f"⚠️ Warnung: Feature '{col}' vom OLS-Modell erwartet, aber nicht in X_test_df. Wird mit Nullen hinzugefügt.")
                        X_test_df[col] = 0
                # Ensure X_test_df has only and exactly the columns in train_columns, in that order
                X_test_df = X_test_df[train_columns]
            else:
                # This case should ideally not happen if the OLS model was trained correctly with feature names
                print("⚠️ Warnung: OLS-Modellinstanz hat kein 'feature_names_in_'. Vorhersage könnte unzuverlässig sein.")
            
            df_results[model_name] = model_instance.predict(X_test_df)
        elif "fpnet" in model_name:
            df_results[model_name] = model_instance.predict(d_test).detach().numpy()
        else:
            print(f"⚠️ Unbekanntes Modell: {model_name}")

    return df_results


def get_table_pvalues(trained_models, d_test, data_type="sim"):
    models_names = get_models_names(trained_models)
    df_results = create_table(models_names, 1)
    for model in models_names:
        if data_type == "sim":
            df_results.loc[:, model["name"]] = model["model"].evaluate_policy(d_test[0])
        elif data_type == "real":
            nuisance_test = model["model"].tarnet.predict_nuisance(d_test.data)
            df_results.loc[:, model["name"]] = model["model"].evaluate_policy(d_test, oracle=False,
                                                                              nuisance=nuisance_test,
                                                                              m=model["model"].m)
        elif data_type == "real_staff":
            nuisance_test = model["model"].tarnet.predict_nuisance(d_test.data)
            df_results.loc[:, model["name"]] = model["model"].evaluate_policy(d_test, oracle=False,
                                                                              nuisance=nuisance_test,
                                                                              m=model["model"].m)    
    return df_results

def get_table_pvalues_conditional(trained_models, d_test, data_type="sim", covariate_cols=None):

    models_names = get_models_names(trained_models)
    df_results = None

    if data_type == "sim":
        df_results = create_table(models_names, len(d_test))
        for model in models_names:
            try:
                result = model["model"].evaluate_policy_perturbed(d_test)

                if result is not None:
                    result_length = len(result)
                    # DataFrame anpassen, falls die Länge der Ergebnisse größer ist als die aktuelle Länge
                    if df_results.shape[0] < result_length:
                        df_results = df_results.reindex(range(result_length))

                    # Ergebnisse direkt einfügen
                    df_results.loc[:result_length - 1, model["name"]] = result

                else:
                    print(f"⚠️ Keine gültigen Ergebnisse für Modell {model['name']}")
            except Exception as e:
                print(f"❌ Fehler beim Bewerten von {model['name']}: {e}")

    elif data_type == "real":
        df_results = create_table(models_names, 2)
        for model in models_names:
            try:
                result = model["model"].evaluate_conditional_pvalues(d_test, oracle=False)

                if result is not None:
                    result_length = len(result)
                    if df_results.shape[0] < result_length:
                        df_results = df_results.reindex(range(result_length))

                    df_results.loc[:result_length - 1, model["name"]] = result
                else:
                    print(f"⚠️ Keine gültigen Ergebnisse für Modell {model['name']}")
            except Exception as e:
                print(f"❌ Fehler beim Bewerten von {model['name']}: {e}")

    elif data_type == "real_staff" or data_type == "job_corps":
        max_length = 0
        results_dict = {}

        for model_info in models_names:
            model_instance = model_info["model"]
            model_name_str = model_info["name"]
            try:
                if model_name_str == "ols":
                    X_test_tensor = d_test.data["x"] # Shape (n_samples, n_covariates) e.g., (1386, 16)
                    y_test_tensor = d_test.data["y"]
                    s_test_tensor = d_test.data["s"]
                    a_test_tensor = d_test.data["a"] # Assignment tensor

                    if covariate_cols is None:
                        raise ValueError("covariate_cols (Liste der Namen der Kovariaten) muss für OLS-Modell in get_table_pvalues_conditional bereitgestellt werden.")

                    # 1. Erstelle DataFrame X_test_df nur mit den Kovariaten-Spalten
                    X_test_df = pd.DataFrame(X_test_tensor.cpu().numpy(), columns=covariate_cols)
                    
                    # 2. Füge die 'assignment'-Spalte hinzu
                    X_test_df["assignment"] = a_test_tensor.cpu().numpy().ravel()
                    
                    # 3. Füge den Interaktionsterm hinzu, falls die Basis-Features vorhanden sind
                    if "trainy1" in X_test_df.columns and "trainy2" in X_test_df.columns:
                        X_test_df["trainy1_x_trainy2"] = X_test_df["trainy1"] * X_test_df["trainy2"]
                    
                    # 4. Richte den DataFrame an den Features aus, mit denen das OLS-Modell trainiert wurde
                    model_trained_features = getattr(model_instance, "feature_names_in_", None)
                    if model_trained_features is not None:
                        model_trained_features = list(model_trained_features) # Sicherstellen, dass es eine Liste ist
                        
                        # Füge fehlende Spalten hinzu, die das Modell erwartet (z.B. Interaktionsterm)
                        for col in model_trained_features:
                            if col not in X_test_df.columns:
                                print(f"⚠️ Warnung (pvalues_conditional OLS): Feature '{col}' vom Modell erwartet, aber nicht in X_test_df. Wird mit Nullen hinzugefügt.")
                                X_test_df[col] = 0
                        
                        # Stelle sicher, dass X_test_df genau die Spalten in model_trained_features hat und in dieser Reihenfolge
                        try:
                            X_test_df = X_test_df[model_trained_features]
                        except KeyError as e:
                            print(f"❌ Fehler beim Anpassen der Spalten für OLS in pvalues_conditional: {e}")
                            print(f"   X_test_df Spalten: {X_test_df.columns.tolist()}")
                            print(f"   Erwartete Spalten: {model_trained_features}")
                            raise
                    else:
                        # Dieser Fall sollte nicht eintreten, wenn das OLS-Modell korrekt mit Feature-Namen trainiert wurde
                        print("⚠️ Warnung (pvalues_conditional OLS): OLS-Modellinstanz hat kein 'feature_names_in_'. Vorhersage könnte unzuverlässig sein.")
                        # Als Fallback könnten die aktuellen Spalten von X_test_df verwendet werden, aber das ist riskant.
                        # Es ist besser, hier einen Fehler auszulösen oder eine robustere Fallback-Logik zu implementieren,
                        # falls dieser Zustand erwartet wird. Fürs Erste wird eine Warnung ausgegeben.

                    y_test_series = pd.Series(y_test_tensor.cpu().numpy().ravel())
                    s_test_series = pd.Series(s_test_tensor.cpu().numpy().ravel())

                    # Die Methode evaluate_conditional_pvalues des OLS-Modells erwartet X_test als DataFrame
                    result = model_instance.evaluate_conditional_pvalues(X_test_df, y_test_series, s_test_series)
                else: # For fpnet and other models expecting Static_Dataset
                    result = model_instance.evaluate_conditional_pvalues(d_test, oracle=False)
                
                if result is not None:
                    result_length = len(result)
                    results_dict[model_name_str] = result
                    max_length = max(max_length, result_length)
                else:
                    print(f"⚠️ Keine gültigen Ergebnisse für Modell {model_name_str}")
                    results_dict[model_name_str] = [np.nan] 
            except Exception as e:
                print(f"❌ Fehler beim Bewerten von {model_name_str} ({data_type}): {e}")
                results_dict[model_name_str] = [np.nan] 

        # DataFrame auf die maximale Ergebnislänge anpassen
        if max_length > 0:
            df_results = create_table(models_names, max_length)
            for model_name, result in results_dict.items():
                df_results.loc[:len(result) - 1, model_name] = result

    else:
        print(f"❌ Unbekannter data_type: {data_type}")

    if df_results is None:
        print("⚠️ Kein gültiges Ergebnis - leeres DataFrame wird zurückgegeben.")
        df_results = pd.DataFrame()  # Fallback für leeres Ergebnis

    # print(f"➡️ Ergebnis-DataFrame:\n{df_results}")  # Ergebnis anzeigen

    return df_results


def get_table_pvalues_max(trained_models, d_test, data_type="sim"):
    models_names = get_models_names(trained_models)
    df_results = create_table(models_names, 1)
    for model in models_names:
        df_results.loc[:, model["name"]] = model["model"].evaluate_worst_case(d_test)
    return df_results


def get_table_action_fairness(trained_models, d_test, data_type="sim"):
    models_names = get_models_names(trained_models)
    df_results = create_table(models_names, 1)
    for model in models_names:
        if data_type == "sim":
            result = model["model"].evaluate_action_fairness(d_test)
        elif data_type == "real":
            s = np.squeeze(d_test.data["s"][:, 1].detach().numpy())
            pi_hat = np.squeeze(model["model"].predict(d_test).detach().numpy())
            test = spearmanr(a=s, b=pi_hat)
            test = test.correlation
            result = np.abs(test)
        elif data_type == "real_staff" or data_type == "job_corps": 
            # Assuming s is the sensitive attribute for fairness, and you want a specific column if it's one-hot encoded.
            # Example: using the second column of s if it represents 'female' after one-hot encoding.
            # Adjust s_column_index as needed. If s is already 1D, just use d_test.data["s"]
            s_column_index = 1 # Example: if 'female' is the second column in your sensitive attributes
            if d_test.data["s"].ndim > 1 and d_test.data["s"].shape[1] > s_column_index:
                s_tensor = d_test.data["s"][:, s_column_index]
            else: # if s is already 1D or has only one column
                s_tensor = d_test.data["s"]
            
            s_np = np.squeeze(s_tensor.cpu().numpy())

            if model["name"] == "ols":
                X_test_tensor = d_test.data["x"]
                feature_names = getattr(model["model"], "feature_names_in_", None)
                if feature_names is None:
                    feature_names = [str(i) for i in range(X_test_tensor.shape[1])]
                
                X_test_df = pd.DataFrame(X_test_tensor.cpu().numpy(), columns=feature_names)
                
                # HINZUFÜGEN: Interaktionsterm erzeugen (wenn nötig)
                if "trainy1" in X_test_df.columns and "trainy2" in X_test_df.columns:
                    X_test_df["trainy1_x_trainy2"] = X_test_df["trainy1"] * X_test_df["trainy2"]
                
                # Prüfe, ob alle Trainings-Features vorhanden sind
                if hasattr(model["model"], "feature_names_in_") and model["model"].feature_names_in_ is not None:
                    missing_features = [f for f in model["model"].feature_names_in_ if f not in X_test_df.columns]
                    if missing_features:
                        print(f"⚠️ Fehlende Features in action_fairness: {missing_features}")
                        for feature in missing_features:
                            X_test_df[feature] = 0
                    X_test_df = X_test_df[model["model"].feature_names_in_]
                
                pi_hat_series = model["model"].predict(X_test_df)
                pi_hat = np.squeeze(pi_hat_series.values)
            else: # For fpnet models
                pi_hat_tensor = model["model"].predict(d_test) # d_test is Static_Dataset
                pi_hat = np.squeeze(pi_hat_tensor.cpu().detach().numpy())
            
            # Ensure s_np and pi_hat are 1D arrays of the same length
            if s_np.ndim > 1: s_np = s_np.ravel()
            if pi_hat.ndim > 1: pi_hat = pi_hat.ravel()

            if len(s_np) == len(pi_hat) and len(s_np) > 1:
                correlation, p_value = spearmanr(a=s_np, b=pi_hat)
                result = np.abs(correlation)
            else:
                print(f"⚠️ Spearmanr warning for {model['name']}: s_np (len {len(s_np)}) and pi_hat (len {len(pi_hat)}) have incompatible shapes or too few elements.")
                result = np.nan # Or 0, or handle as error
        df_results.loc[:, model["name"]] = result
    return df_results


def get_table_action_fairness_repr(trained_models, d_test):
    models_names = get_models_names(trained_models)
    df_results = create_table(models_names, 1)
    for model in models_names:
        df_results.loc[:, model["name"]] = model["model"].evaluate_action_fairness_repr(d_test)
    return df_results


def get_table_reconstruction_repr(trained_models, d_test, sensitive_feat=1):
    data = d_test[0].data
    x = data["x"]
    dim_x = x.shape[1]
    x_us = x[:, 0:dim_x - sensitive_feat].detach().numpy()
    x_s = x[:, dim_x - sensitive_feat:].detach().numpy()
    models_names = get_models_names(trained_models)
    df_results = create_table(models_names, 2)
    for model in models_names:
        if hasattr(model["model"], "repr_net"):
            if model["model"].repr_net is not None:
                x_hat = model["model"].repr_net.predict_reconstruction(data).detach().numpy()
                x_hat_us = x_hat[:, 0:dim_x - sensitive_feat]
                x_hat_s = x_hat[:, dim_x - sensitive_feat:]
                df_results.loc[0, model["name"]] = np.mean(np.sqrt(np.sum((x_hat_us - x_us) ** 2, axis=1)))
                df_results.loc[1, model["name"]] = np.mean(np.sqrt(np.sum((x_hat_s - x_s) ** 2, axis=1)))
            else:
                df_results.loc[:, model["name"]] = None
        else:
            df_results.loc[:, model["name"]] = None
    return df_results


# Plot policies for binary S, dim x_us, x_s = 1
def plot_policies1D(trained_models, d_test):
    data_test = d_test[0]
    x_us_test = data_test.data["x"][:, 0:1]
    n_test = x_us_test.shape[0]
    ite_f = data_test.nuisance["mu1_f"] - data_test.nuisance["mu0_f"]
    # Plot ITEs for both sensitive groups
    ite_1 = data_test.nuisance["mu1_s1"] - data_test.nuisance["mu0"]
    ite_0 = data_test.nuisance["mu1_s0"] - data_test.nuisance["mu0"]
    ite_1 = ite_1.detach().numpy()
    ite_0 = ite_0.detach().numpy()
    ite_f = ite_f.detach().numpy()
    data = np.concatenate((x_us_test.detach().numpy(), ite_1, ite_0, ite_f), axis=1)
    models_names = get_models_names(trained_models)
    for model in models_names:
        predictions = model["model"].predict(data_test)
        data = np.concatenate((data, predictions.detach().numpy()), axis=1)
    data = data[data[:, 0].argsort()]
    plt.plot(data[:, 0], data[:, 1], label=r"$\ite1$", color="mediumblue")
    plt.plot(data[:, 0], data[:, 2], label=r"$\ite0$", color="orchid")
    plt.plot(data[:, 0], data[:, 3], label=r"$\itef$", color="lime")
    colors = ["orange", "darkred", "deepskyblue", "darkblue"]
    for i, model in enumerate(models_names):
        plt.plot(data[:, 0], data[:, 4 + i], label=model["name"], color=colors[i])
    plt.legend()
    plt.show()