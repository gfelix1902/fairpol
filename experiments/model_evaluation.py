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


def get_policy_predictions(trained_models, d_test, data_type="real"):
    models_names = get_models_names(trained_models)
    df_results = pd.DataFrame(columns=[model["name"] for model in models_names], index=range(d_test.data["y"].shape[0]))

    for model in models_names:
        model_name = model["name"]
        model_instance = model["model"]

        if model_name == "ols":
            X_test_tensor = d_test.data["x"]
            X_test_df = pd.DataFrame(X_test_tensor.cpu().numpy())
            X_test_df["assignment"] = d_test.data["a"].cpu().numpy().ravel()
            X_test_df.columns = [str(col) for col in X_test_df.columns]

            train_columns = getattr(model_instance, "feature_names_in_", None)
            if train_columns is not None:
                train_columns = list(train_columns)
                for col in train_columns:
                    if col not in X_test_df.columns:
                        X_test_df[col] = 0
                X_test_df = X_test_df[train_columns]
            else:
                print("⚠️ Konnte Trainingsspalten nicht abrufen (feature_names_in_ fehlt im Modell).")
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

def get_table_pvalues_conditional(trained_models, d_test, data_type="sim"):

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
                    X_test_tensor = d_test.data["x"]
                    y_test_tensor = d_test.data["y"]
                    s_test_tensor = d_test.data["s"]

                    X_test_df = pd.DataFrame(X_test_tensor.cpu().numpy())
                    X_test_df["assignment"] = d_test.data["a"].cpu().numpy().ravel()
                    X_test_df.columns = [str(col) for col in X_test_df.columns]

                    train_columns = getattr(model_instance, "feature_names_in_", None)
                    if train_columns is not None:
                        train_columns = list(train_columns)
                        for col in train_columns:
                            if col not in X_test_df.columns:
                                X_test_df[col] = 0
                        X_test_df = X_test_df[train_columns]
                    else:
                        print("⚠️ Konnte Trainingsspalten nicht abrufen (feature_names_in_ fehlt im Modell).")

                    y_test_series = pd.Series(y_test_tensor.cpu().numpy().ravel())
                    s_test_series = pd.Series(s_test_tensor.cpu().numpy().ravel())

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
        elif data_type == "real_staff" or data_type == "job_corps": # Added job_corps
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
                X_test_df = pd.DataFrame(X_test_tensor.cpu().numpy())
                pi_hat_series = model["model"].predict(X_test_df)
                pi_hat = np.squeeze(pi_hat_series.values) # .values to get NumPy array from Series
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