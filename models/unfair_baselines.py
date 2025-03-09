import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as fctnl
import utils
from models.abstract import PolicyNet
from models.policy_scores import policy_value


# TARNet for nuisance parameter estimation
def train_tarnet(datasets, config):
    tarnet = TARNet(config)
    train_list = utils.train_model(tarnet, datasets, config)
    return train_list


class TARNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        input_size = config["data"]["x_dim"] + config["data"]["s_dim"]
        hidden_size_1 = config["model"]["hidden_size_1"]
        hidden_size_2 = config["model"]["hidden_size_2"]
        hidden_size_prop = config["model"]["hidden_size_prop"]
        dropout = config["model"]["dropout"]

        self.body = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_1, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_1, hidden_size_1),
            nn.ReLU(),
        )

        self.head1 = nn.Sequential(
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_2, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_2, 1),
        )

        self.head0 = nn.Sequential(
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_2, hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_2, 1),
        )

        self.prophead = nn.Sequential(
            nn.Linear(input_size, hidden_size_prop),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_prop, hidden_size_prop),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_prop, 1),
            nn.Sigmoid(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["model"]["lr"])
        self.y_type = config["data"]["y_type"]
        self.neptune = config["experiment"]["neptune"]
        self.save_hyperparameters(config["model"])

    def configure_optimizers(self):
        return self.optimizer

    def objective(self, batch, out):
        # prediction
        y_hat = batch["a"] * out["mu1"] + (1 - batch["a"]) * out["mu0"]
        loss_mu = utils.mse_bce(y=batch["y"], y_hat=y_hat, y_type=self.y_type)
        loss_prop = fctnl.binary_cross_entropy(out["prop"], batch["a"], reduction='mean')
        # Final loss
        loss = loss_prop + loss_mu
        return {"obj": loss, "loss_prop": loss_prop, "loss_mu": loss_mu}

    def forward(self, batch):
        x = batch["x"]
        s = batch["s"]
        cov = torch.concat([x, s], dim=1)
        tar_body = self.body(cov)
        mu1_hat = self.head1(tar_body)
        mu0_hat = self.head0(tar_body)
        if self.y_type == "binary":
            mu1_hat = torch.sigmoid(mu1_hat)
            mu0_hat = torch.sigmoid(mu0_hat)
        prop = self.prophead(cov)
        return {"mu1": mu1_hat, "mu0": mu0_hat, "prop": prop}

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Forward pass
        out = self.forward(train_batch)
        # Loss
        loss_dict = self.objective(train_batch, out)
        # Logging
        if self.neptune:
            self.log_dict(loss_dict, logger=True, on_epoch=True, on_step=False)
        return loss_dict["obj"]

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        # Forward pass
        out = self.forward(train_batch)
        # Loss
        loss_dict = self.objective(train_batch, out)
        #Validation accuracy
        prop_hat = out["prop"]
        a_pred = torch.heaviside(prop_hat - 0.5, torch.zeros((prop_hat.size(0), 1), device=prop_hat.device))
        a_target = train_batch["a"]
        acc_val = torch.sum(a_pred == a_target) / prop_hat.size(0)
        loss_dict_val = dict([("val_" + key, value) for key, value in loss_dict.items()])
        loss_dict_val["val_acc"] = acc_val
        # Logging
        self.log_dict(loss_dict_val, logger=True, on_epoch=True, on_step=False)
        return loss_dict_val["val_obj"]

    def predict_nuisance(self, batch):
        self.eval()
        nuisance_dict = self.forward(batch)
        nuisance_dict["mu1"] = nuisance_dict["mu1"].detach()
        nuisance_dict["mu0"] = nuisance_dict["mu0"].detach()
        nuisance_dict["prop"] = nuisance_dict["prop"].detach()
        return nuisance_dict


class OraclePolicy(PolicyNet):
    def __init__(self):
        super().__init__()

    def predict(self, data_test):
        n_test = data_test.data["y"].shape[0]
        mu1 = data_test.nuisance["mu1"]
        mu0 = data_test.nuisance["mu0"]
        ite = mu1 - mu0
        pi_hat = torch.heaviside(ite, torch.zeros((n_test, 1)))
        return pi_hat


class OraclePolicy_af(PolicyNet):
    def __init__(self):
        super().__init__()

    def predict(self, data_test):
        n_test = data_test.data["y"].shape[0]
        mu1_f = data_test.nuisance["mu1_f"]
        mu0_f = data_test.nuisance["mu0_f"]
        ite_f = mu1_f - mu0_f
        pi_hat = torch.heaviside(ite_f, torch.zeros((n_test, 1)))
        return pi_hat
