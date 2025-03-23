import os
import yaml
import torch
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pytorch_lightning.loggers.neptune import NeptuneLogger
import torch.nn.functional as fctnl
from torch.utils.data import DataLoader
import random
import numpy as np
from data.sim_binary_s2 import generate_datasets
from data.load_real import load_oregon
from data.load_real_staff import main as load_staff_data
from data.load_real_job import load_data_from_csv

def get_device():
    if torch.cuda.is_available():
        gpu = 1
    else:
        gpu = 0
    return gpu

def get_device_string():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_project_path():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    return str(path.absolute())

def load_yaml(path_relative):
    print(f"Lade YAML-Konfigurationsdatei: {path_relative}.yaml")
    try:
        config = yaml.safe_load(open(get_project_path() + path_relative + ".yaml", 'r'))
        print("YAML-Konfigurationsdatei erfolgreich geladen.")
        return config
    except FileNotFoundError:
        print(f"Fehler: YAML-Konfigurationsdatei {path_relative}.yaml nicht gefunden.")
        return None
    except yaml.YAMLError as e:
        print(f"Fehler beim Laden der YAML-Konfigurationsdatei: {e}")
        return None

def save_yaml(path_relative, file):
    with open(get_project_path() + path_relative + ".yaml", 'w') as outfile:
        yaml.dump(file, outfile, default_flow_style=False)

def load_data(config_data, standardize=True):
    print(f"Lade Datensatz: {config_data['dataset']}")
    if config_data["dataset"] == "sim":
        datasets = generate_datasets(config_data)
        print("Simulierter Datensatz erfolgreich geladen.")
        return datasets
    elif config_data["dataset"] == "real":
        datasets = load_oregon(config_data, standardize=standardize)
        print("Oregon Datensatz erfolgreich geladen.")
        return datasets
    elif config_data["dataset"] == "real_staff" or config_data["dataset"] == "job_corps":
        # datasets = load_staff_data(config_data)
        datasets = load_data_from_csv(config_data)
        if datasets:
            print(f"{config_data['dataset']} Datensatz erfolgreich geladen.")
            return datasets
        else:
            print(f"Fehler beim Laden des {config_data['dataset']} Datensatzes.")
            return None
    else:
        raise ValueError(f"Unknown dataset: {config_data['dataset']}")

def get_logger(neptune=True):
    if neptune:
        logger = True
    else:
        logger = True
    return logger

def get_config_names(model_configs):
    config_names = [model_config["name"] for model_config in model_configs]
    return config_names

def get_config_af(model_configs):
    config_af = [model_config["action_fair"] for model_config in model_configs if "action_fair" in model_config]
    return config_af

def plot_TSNE_repr(psi, phi):
    n = psi.shape[0]
    tsne = TSNE(n_components=1, n_iter=300)
    embedd1 = tsne.fit_transform(psi)
    embedd2 = tsne.fit_transform(phi)
    df_plot = pd.DataFrame(columns=['x', 'y'], index=list(range(n)))
    df_plot.iloc[:n, 0:1] = embedd1
    df_plot.iloc[:n, 1:2] = embedd2
    plt.plot(df_plot.x, df_plot.y, marker='o', linestyle='', markersize=5, alpha=0.5)
    plt.xlabel("Fair")
    plt.ylabel("Sensitive")
    plt.xlim((-10, 10))
    plt.title("TSNE of representations")
    plt.show()

def plot_TSNE_repr_label(repr, label, binary=True, title="TSNE of representations"):
    if binary:
        label = label[:, 0]
    n = repr.shape[0]
    tsne = TSNE(n_components=2, n_iter=300)
    embedd1 = tsne.fit_transform(repr)
    df_plot = pd.DataFrame(columns=['x', 'y'], index=list(range(n)))
    df_plot.iloc[:, 0:2] = embedd1
    plt.scatter(df_plot.x, df_plot.y, marker='o', c=label)
    plt.title(title)
    plt.show()

def mse_bce(y, y_hat, y_type="continuous"):
    if y_type == "continuous":
        return torch.mean((y - y_hat) ** 2)
    if y_type == "binary":
        return fctnl.binary_cross_entropy(y_hat, y, reduction='mean')

def train_model(model, datasets, config):
    print("Starte Modelltraining...")
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    validation = config["experiment"]["validation"]
    logger = get_logger(config["experiment"]["neptune"])

    # trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False,
    #                      devices=1 if get_device() else 0, accelerator="gpu" if get_device() else "cpu", logger=logger, enable_checkpointing=False)

    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="gpu" if get_device() else "cpu",
        logger=logger,
        enable_checkpointing=False
    )

    train_loader = DataLoader(dataset=datasets["d_train"], batch_size=batch_size, shuffle=True)
    try:
        if validation:
            val_loader = DataLoader(dataset=datasets["d_val"], batch_size=batch_size, shuffle=False)
            trainer.fit(model, train_loader, val_loader)
            val_results = trainer.validate(model=model, dataloaders=val_loader, verbose=False)
        else:
            trainer.fit(model, train_loader)
            val_results = None
        print("Modelltraining abgeschlossen.")
        return {"trained_model": model, "val_results": val_results[0], "logger": logger}
    except Exception as e:
        print(f"Fehler beim Modelltraining: {e}")
        return None