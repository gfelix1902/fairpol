from data.data_structures import Static_Dataset
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
import random
import utils


def generate_datasets(config, to_tensor=True):
    n_train = config["n_train"]
    n_val = config["n_val"]
    n_test = config["n_test"]
    prob_s0 = config["prob_s0"]
    # Trainin set
    S_train = get_random_S(n_train, prob_s0)
    enc = OneHotEncoder().fit(S_train)
    data_train = simulate_data(n_train, config, S_train, enc)
    # Validation set
    data_val = simulate_data(n_val, config, get_random_S(n_val, prob_s0), enc)
    # Different test datasets (interventions on s)
    S_test = get_random_S(n_test, prob_s0)
    new_seed = random.randint(0, 1000000)
    utils.set_seed(new_seed)
    data_test = [simulate_data(n_test, config, S_test, enc)]
    utils.set_seed(new_seed)
    data_test.append(simulate_data(n_test, config, np.full(shape=(n_test, 1), fill_value=0), enc))
    utils.set_seed(new_seed)
    data_test.append(simulate_data(n_test, config, np.full(shape=(n_test, 1), fill_value=1), enc))

    if to_tensor:
        data_train.convert_to_tensor()
        data_val.convert_to_tensor()
        for d_test in data_test:
            d_test.convert_to_tensor()

    return {"d_train": data_train, "d_val": data_val, "d_test": data_test}


def simulate_data(n, config, S, one_hot_encoder):
    p_xu = config["p_xu"]
    p_xs = config["p_xs"]
    noise_Y = config["noise_Y"]
    prob_s0 = config["prob_s0"]

    # Non-sensitive features correlated with S (noisy proxies)
    X_s = np.random.uniform(low=0, high=1, size=(n, p_xs)) + (S - 1)
    # Independent non-sensitive features
    X_us = np.random.uniform(low=-1, high=1, size=(n, p_xu))
    X = np.concatenate([X_us, X_s], axis=1)
    confounder = np.concatenate([X, S], axis=1)

    # create propensity scores and treatment
    feat = np.sin(2 * confounder)
    prob = expit(np.sum(feat, axis=1)) * 0.8 + 0.1
    A = np.expand_dims(np.random.binomial(n=1, p=prob, size=n), axis=1)
    #Create ITEs
    ite_base = np.where(np.mean(X_us, axis=1) < 0.5, 0.3 * np.mean(np.sin(4 * X_us - 2), axis=1), 0) #+ 0.3
    ite = ite_base + np.where(np.mean(X_us, axis=1) > 0.5, 0.6 * np.squeeze(S) - 0.3, 0) #+ np.where(np.mean(X_us, axis=1) < 0, -0.4 * np.squeeze(S) + 0.2, 0)
    expectation_s = 1 - prob_s0
    ite_f = ite_base + np.where(np.mean(X_us, axis=1) > 0.5, 0.6 * expectation_s - 0.3, 0) #+ np.where(np.mean(X_us, axis=1) < 0, -0.4 * expectation_s + 0.2, 0)
    ite_s1 = ite_base + np.where(np.mean(X_us, axis=1) > 0.5, 0.6 * 1 - 0.3, 0) #+ np.where(np.mean(X_us, axis=1) < 0, -0.4 * 1 + 0.2, 0)
    ite_s0 = ite_base + np.where(np.mean(X_us, axis=1) > 0.5, 0.6 * 0 - 0.3, 0) #+ np.where(np.mean(X_us, axis=1) < 0, -0.4 * 0 + 0.2, 0)
    # Create outcome functions
    mu0 = np.zeros(n)
    mu1 = mu0 + ite
    mu1_f = mu0 + ite_f
    mu0_f = mu0
    mu1_s1 = mu0 + ite_s1
    mu1_s0 = mu0 + ite_s0
    #Expand dimensions
    mu1 = np.expand_dims(mu1, axis=1)
    mu0 = np.expand_dims(mu0, axis=1)
    mu1_f = np.expand_dims(mu1_f, axis=1)
    mu0_f = np.expand_dims(mu0_f, axis=1)
    mu1_s1 = np.expand_dims(mu1_s1, axis=1)
    mu1_s0 = np.expand_dims(mu1_s0, axis=1)

    #Create outcomes
    Y = A * mu1 + (1 - A) * mu0 + np.random.normal(0, scale=noise_Y, size=(n, 1))

    # One-hot encoding for binary S
    s_one_hot = one_hot_encoder.transform(S).toarray()
    dataset = Static_Dataset(y=Y, a=A, x=X, s=s_one_hot, y_type="continuous", x_type=["continuous"] * (p_xu + p_xs),
                             dim_xu=p_xu,
                             s_type=["categorical"], mu1=mu1, mu0=mu0, prop=np.expand_dims(prob, 1), mu1_f=mu1_f,
                             mu0_f=mu0_f,
                             mu1_s1=mu1_s1, mu1_s0=mu1_s0)
    #Standardize outcomes
    dataset.scale_y()
    return dataset


def get_random_S(n, prob_s0):
    S_train = np.random.choice([0, 1], size=(n, 1), p=[prob_s0, 1 - prob_s0])
    return S_train
