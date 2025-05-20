import itertools
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as fctnl
from torch.utils.data import DataLoader
import utils
from torch.utils.data import Dataset
from models.policy_scores import score_dm, policy_value
from models.abstract import PolicyNet
import numpy as np
import ot

#Training of fair representation
def train_fair_repr(datasets, config_repr, loss="wstein"):
    if loss == "wstein":
        repr_model = ReprNetWstein(config_repr, s=datasets["d_train"].data["s"])
    if loss == "conf":
        repr_model = ReprNetConfusion(config_repr)
    if loss == "gr":
        repr_model = ReprNetGR(config_repr)
    return utils.train_model(repr_model, datasets, config_repr)


#Parent class for representation net
class ReprNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        hidden_size_phi = config["model"]["hidden_size_phi"]
        input_size = config["data"]["x_dim"]
        dropout = config["model"]["dropout"]
        self.phi_net = nn.Sequential(
            nn.Linear(input_size, hidden_size_phi),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_phi, hidden_size_phi),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_phi, hidden_size_phi),
            nn.Tanh(),
        )

        self.reconstruction = nn.Sequential(
            nn.Linear(hidden_size_phi + config["data"]["s_dim"], hidden_size_phi),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_phi, hidden_size_phi),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_phi, 1),
        )
        self.neptune = config["experiment"]["neptune"]
        self.gamma = config["model"]["gamma"]

    def log_dict_prefix(self, loss_dict, prefix=""):
        # Rename keys with prefix
        loss_dict_prefix = dict([(prefix + "_" + key, value) for key, value in loss_dict.items()])
        self.log_dict(loss_dict_prefix, logger=True, on_epoch=True, on_step=False)

    def predict_repr(self, data):
        self.eval()
        phi = self.phi_net(data["x"]).detach()
        return phi

    def predict_reconstruction(self, data):
        x = data["x"]
        s = data["s"]
        phi = self.phi_net(x)
        x_hat = self.reconstruction(torch.concat([phi, s], dim=1))
        return x_hat.detach()


# Wasserstein loss---------------------------------------------------------------------------------
class ReprNetWstein(ReprNet):
    def __init__(self, config, s):
        super().__init__(config)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["model"]["lr"],
                                          weight_decay=config["model"]["weight_decay"])
        self.s_categories, self.s_counts = torch.unique(s, dim=0, return_counts=True)
        self.save_hyperparameters(config["model"])

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, batch):
        x = batch["x"]
        s = batch["x"]
        phi = self.phi_net(x)
        x_hat = self.reconstruction(torch.concat([phi, s], dim=1))
        return {"phi": phi, "x_hat": x_hat}

    def get_repr_by_s(self, batch, phi):
        s = batch["s"]
        batch_size = s.size(0)
        data_list = []
        for i in range(self.s_categories.size(0)):
            index = []
            for j in range(batch_size):
                if torch.equal(s[j, :], self.s_categories[i, :]):
                    index.append(j)
            data_list.append(phi[index, :])
        return data_list

    def obj(self, batch, out):
        # Reconstruction loss
        x = batch["x"]
        x_hat = out["x_hat"]
        loss_reconstruction = utils.mse_bce(x, x_hat)
        # Wasserstein regularization
        repr_by_s = self.get_repr_by_s(batch, out["phi"])
        M = ot.dist(repr_by_s[0], repr_by_s[1])
        a = torch.ones(M.size(0), device=M.device) / M.size(0)
        b = torch.ones(M.size(1), device=M.device) / M.size(1)
        loss_wstein = ot.emd2(a=a, b=b, M=M)  # - torch.mean(M)

        # Normalize wasserstein loss
        repr = torch.concat([repr_by_s[0], repr_by_s[1]], dim=0)
        repr_centered = repr - torch.tile(torch.mean(repr, dim=0), (x.size(0), 1))
        repr_norm = torch.mean(torch.sqrt(torch.sum(torch.square(repr - repr_centered), dim=1)))
        # repr_norm = torch.mean(ot.dist(repr, repr))
        loss_wstein_scaled = loss_wstein / repr_norm
        if self.current_epoch > 50:
            pass
        # schedule = (2 / (1 + math.exp(-0.01 * (self.current_epoch - 30)))) - 1
        # if self.current_epoch >= 70:
        #    obj = loss_wstein #* schedule
        # else:
        #    obj = 0
        obj = loss_reconstruction + self.gamma * loss_wstein_scaled
        return {"obj": obj, "loss_wstein": loss_wstein, "schedule": 0, "loss_reconstruction": loss_reconstruction,
                "repr_norm": repr_norm, "loss_wstein_scaled": loss_wstein_scaled}

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Step 1
        out = self.forward(train_batch)
        loss_dict = self.obj(train_batch, out)
        # Logging
        if self.neptune:
            self.log_dict_prefix(loss_dict, "train")
        return loss_dict["obj"]

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        # Step 1
        out = self.forward(train_batch)
        loss_dict = self.obj(train_batch, out)
        # Logging
        self.log_dict_prefix(loss_dict, "val")
        return loss_dict["obj"]


# Domain Confusion loss---------------------------------------------------------------------------------
class ReprNetConfusion(ReprNet):
    def __init__(self, config):
        super().__init__(config)
        hidden_size_phi = config["model"]["hidden_size_phi"]
        dropout = config["model"]["dropout"]
        s_size = config["data"]["s_dim"]
        self.s_adv = nn.Sequential(
            nn.Linear(hidden_size_phi, hidden_size_phi),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_phi, hidden_size_phi),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_phi, s_size),
            nn.Softmax()
        )
        # Optimization
        self.automatic_optimization = False
        model_parameters = list(self.phi_net.parameters()) + list(self.reconstruction.parameters())
        self.optimizer_model = torch.optim.Adam(model_parameters, lr=config["model"]["lr"],
                                                weight_decay=config["model"]["weight_decay"])
        self.optimizer_adv = torch.optim.Adam(self.s_adv.parameters(), lr=config["model"]["lr"],
                                              weight_decay=config["model"]["weight_decay"])
        self.save_hyperparameters(config["model"])

    def configure_optimizers(self):
        return self.optimizer_model, self.optimizer_adv

    def forward(self, batch):
        phi = self.phi_net(batch["x"])
        s_hat = self.s_adv(phi)
        x_hat = self.reconstruction(torch.concat([phi, batch["s"]], dim=1))
        return {"phi": phi, "s_hat": s_hat, "x_hat": x_hat}

    def obj_1(self, batch, out):
        n = batch["y"].size(0)
        x = batch["x"]
        y_hat = out["x_hat"]
        loss_reconstruction = utils.mse_bce(y_hat, batch["y"])
        uniform_tensor = torch.full(size=(n, batch["s"].size(1)), fill_value=1 / batch["s"].size(1),
                                    device=batch["s"].device)
        loss_confusion = fctnl.cross_entropy(out["s_hat"], uniform_tensor, reduction='mean')
        # schedule = (2 / (1 + math.exp(-0.02 * self.current_epoch))) - 1
        schedule = 1
        obj = self.gamma * schedule * loss_confusion + loss_reconstruction
        return {"obj": obj, "loss_confusion": loss_confusion, "schedule": schedule,
                "loss_reconstruction": loss_reconstruction}

    def obj_2(self, batch, out):
        loss_adv = fctnl.cross_entropy(out["s_hat"], batch["s"], reduction='mean')
        return {"loss_adv": loss_adv}

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Step 1
        out = self.forward(train_batch)
        loss_dict = self.obj_1(train_batch, out)
        # Logging
        if self.neptune:
            self.log_dict_prefix(loss_dict, "train")
        # Optimize step 1
        self.optimizer_model.zero_grad()
        self.manual_backward(loss_dict["obj"])
        self.optimizer_model.step()

        # Step 2
        out_adv = self.forward(train_batch)
        loss_dict_adv = self.obj_2(train_batch, out_adv)
        # Logging
        if self.neptune:
            self.log_dict_prefix(loss_dict_adv, "train")
        # Optimize step 2
        self.optimizer_adv.zero_grad()
        self.manual_backward(loss_dict_adv["loss_adv"])
        self.optimizer_adv.step()

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        out = self.forward(train_batch)
        loss_dict = self.obj_1(train_batch, out)
        # Logging
        self.log_dict_prefix(loss_dict, "val")
        return loss_dict["obj"]

# Gradient reversal loss-----------------------------------------------------
class RevGrad_fcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, lamb):
        ctx.save_for_backward(input_, lamb)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, lamb = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * lamb
        return grad_input, None


class RevGrad(torch.nn.Module):
    def __init__(self):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__()

    def forward(self, input_, lamb):
        revgrad = RevGrad_fcn.apply
        return revgrad(input_, lamb)


class ReprNetGR(ReprNet):
    def __init__(self, config):
        super().__init__(config)
        hidden_size_phi = config["model"]["hidden_size_phi"]
        dropout = config["model"]["dropout"]
        s_size = config["data"]["s_dim"]
        self.s_adv = nn.Sequential(
            nn.Linear(hidden_size_phi, hidden_size_phi),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_phi, hidden_size_phi),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size_phi, s_size),
            nn.Softmax()
        )
        # Optimization
        self.reversal = RevGrad()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["model"]["lr"],
                                          weight_decay=config["model"]["weight_decay"])
        self.save_hyperparameters(config["model"])

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, batch):
        #schedule = (2 / (1 + math.exp(-0.02 * self.current_epoch))) - 1
        schedule = 1
        phi = self.phi_net(batch["x"])
        if self.current_epoch > 200:
            schedule = torch.tensor(self.gamma * schedule, requires_grad=False)
        else:
            schedule = torch.tensor(0, requires_grad=False)
        s_hat = self.s_adv(self.reversal(phi, schedule))
        x_hat = self.reconstruction(torch.concat([phi, batch["s"]], dim=1))
        return {"phi": phi, "s_hat": s_hat, "x_hat": x_hat, "schedule": schedule}

    def obj(self, batch, out):
        x = batch["x"]
        x_hat = out["x_hat"]
        loss_reconstruction = utils.mse_bce(x, x_hat)
        loss_adv = fctnl.cross_entropy(out["s_hat"], batch["s"], reduction='mean')
        obj = loss_adv + loss_reconstruction
        return {"obj": obj, "loss_adv": loss_adv, "schedule": out["schedule"], "loss_reconstruction": loss_reconstruction}

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Step 1
        out = self.forward(train_batch)
        loss_dict = self.obj(train_batch, out)
        # Logging
        if self.neptune:
            self.log_dict_prefix(loss_dict, "train")
        return loss_dict["obj"]

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        out = self.forward(train_batch)
        loss_dict = self.obj(train_batch, out)
        # Logging
        self.log_dict_prefix(loss_dict, "val")
        return loss_dict["obj"]


# Policy net----------------------------------------------------------------------------------

def train_fpnet(datasets, config, tarnet=None, repr_net=None):
    fpnet = FPNet(config, s=datasets["d_train"].data["s"], tarnet=tarnet, repr_net=repr_net)
    if tarnet is None:
        datasets["d_train"].set_load_nuisance(True)
        datasets["d_val"].set_load_nuisance(True)
    if config["experiment"]["nuisance_only"] == False:
        train_list = utils.train_model(fpnet, datasets, config)
    else:
        train_list = {"trained_model": fpnet}

    return train_list

class FPNet(PolicyNet):
    def __init__(self, config, s, tarnet=None, repr_net=None):
        super().__init__()
        hidden_size_pi = config["model"]["hidden_size_pi"]
        dropout = config["model"]["dropout"]
        s_size = config["data"]["s_dim"]
        # Fairness configurations
        self.action_fair = config["model"]["af"]
        self.value_fair = config["model"]["vf"]
        # Nuisance parameters
        self.tarnet = tarnet
        self.repr_net = repr_net
        # Inputs size
        if self.action_fair in ["af_wstein", "af_conf", "af_gr"]:
            input_size = self.repr_net.phi_net[0].out_features
        if self.action_fair == "auf":
            input_size = config["data"]["x_dim"] + s_size
        if self.action_fair == "afo":
            input_size = config["data"]["xu_dim"]

        self.pi_net = nn.Sequential(
            nn.Linear(input_size, hidden_size_pi),
            nn.ELU(),
            nn.Dropout(dropout),
            #nn.Linear(hidden_size_pi, hidden_size_pi),
            #nn.ELU(),
            #nn.Dropout(dropout),
            nn.Linear(hidden_size_pi, 1),
            nn.Sigmoid()
        )

        # Policy score method
        self.m = config["model"]["m"]
        #Budget constraint
        self.delta = config["model"]["delta"]
        # Get sensitive categories and probabilities from whole dataset
        s_categories, s_counts = torch.unique(s, dim=0, return_counts=True)
        s_prob = s_counts / s.size(0)
        self.s_categories = s_categories.to(device=utils.get_device_string())
        self.s_prob = s_prob.to(device=utils.get_device_string())
        if self.value_fair == "vef":
            self.lamb = config["model"]["lamb"]
        self.neptune = config["experiment"]["neptune"]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["model"]["lr"],
                                          weight_decay=config["model"]["weight_decay"])
        self.save_hyperparameters(config["model"])

    def get_input(self, batch):
        if self.action_fair in ["af_wstein", "af_conf", "af_gr"]:
            return self.repr_net.predict_repr(batch)
        if self.action_fair == "auf":
            return torch.concat([batch["x"], batch["s"]], dim=1)
        if self.action_fair == "afo":
            return batch["x"][:, 0:self.pi_net[0].in_features]

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, batch):
        pi_hat = self.pi_net(self.get_input(batch))
        return {"pi_hat": pi_hat}

    def get_nuisance(self, batch):
        if self.tarnet is not None:
            return self.tarnet.predict_nuisance(batch)
        else:
            return {"mu1": batch["mu1"], "mu0": batch["mu0"], "prop": batch["prop"]}

    def budget(self, pi_hat):
        return torch.mean(pi_hat)

    def policy_value(self, batch, pi_hat):
        a = batch["a"]
        y = batch["y"]
        nuisance = self.get_nuisance(batch)
        pi_value = policy_value(pi_hat, mu_1=nuisance["mu1"], mu_0=nuisance["mu0"], prop=nuisance["prop"], a=a, y=y,
                                m=self.m)
        return pi_value

    def get_conditional_pvalues(self, batch, pi_hat):
        nuisance = self.get_nuisance(batch)
        s = batch["s"]
        batch_size = s.size(0)
        pi_values = []
        for i in range(self.s_categories.size(0)):
            index = []
            for j in range(batch_size):
                if torch.equal(s[j, :], self.s_categories[i, :]):
                    index.append(j)
            pi_values.append(
                policy_value(pi_hat[index, :], mu_1=nuisance["mu1"][index, :], mu_0=nuisance["mu0"][index, :],
                             prop=nuisance["prop"][index, :], a=batch["a"][index, :], y=batch["y"][index, :],
                             m=self.m, summary="mean"))
        return pi_values

    def obj_unfair(self, batch, pi_hat):
        pi_value = self.policy_value(batch, pi_hat)
        obj = -pi_value + self.delta * self.budget(pi_hat)
        return {"obj": obj, "pvalue": pi_value, "budget": self.budget(pi_hat)}

    def obj_max_min(self, batch, pi_hat):
        cond_pvalues = self.get_conditional_pvalues(batch, pi_hat)
        worst_case_p_value = min(cond_pvalues)
        min_nr = cond_pvalues.index(worst_case_p_value)
        p_value = self.policy_value(batch, pi_hat)
        obj = -worst_case_p_value + self.delta * self.budget(pi_hat)
        return {"obj": obj, "budget": self.budget(pi_hat), "pvalue0": cond_pvalues[0], "pvalue1": cond_pvalues[1],
                "pvalue_worst_case": worst_case_p_value, "pvalue": p_value, "min_nr": min_nr}

    def obj_envy_free(self, batch, pi_hat):
        # Get maximal difference
        cond_pvalues = self.get_conditional_pvalues(batch, pi_hat)
        index = list(range(len(cond_pvalues)))
        combinations = list(itertools.combinations(index, r=2))
        diffs = []
        for combination in combinations:
            diffs.append(torch.abs(cond_pvalues[combination[0]] - cond_pvalues[combination[1]]))
        max_difference = max(diffs)
        # Get policy value
        pi_value = self.policy_value(batch, pi_hat)
        obj = -pi_value
        if self.current_epoch > 50:
            obj += self.lamb * max_difference + self.delta * self.budget(pi_hat)
        return {"obj": obj, "pvalue": pi_value, "max_difference": max_difference, "budget": self.budget(pi_hat)}

    def obj_policy(self, batch, pi_hat):
        if self.value_fair == "vmm":
            return self.obj_max_min(batch, pi_hat)
        if self.value_fair == "vuf":
            return self.obj_unfair(batch, pi_hat)
        if self.value_fair == "vef":
            return self.obj_envy_free(batch, pi_hat)

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Step 1
        out = self.forward(train_batch)
        loss_dict = self.obj_policy(train_batch, out["pi_hat"])
        # Logging
        if self.neptune:
            self.log_dict_prefix(loss_dict, "train")
        return loss_dict["obj"]

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        # Step 1
        out = self.forward(train_batch)
        loss_dict = self.obj_policy(train_batch, out["pi_hat"])
        # Logging (always for validation)
        self.log_dict_prefix(loss_dict, "val")
        return loss_dict["obj"]

    def predict(self, data):
        self.eval()
        pi_hat = self.pi_net(self.get_input(data.data))
        return pi_hat

    def predict_repr(self, data):
        self.eval()
        phi = self.repr_net.predict_repr(data)
        return phi

    def evaluate_conditional_pvalues(self, data, oracle=False):
        self.eval()
        pi_hat = self.pi_net(self.get_input(data.data))
        if not oracle:
            nuisance_test = self.tarnet.predict_nuisance(data.data)
        else:
            nuisance_test = data.nuisance
        s = data.data["s"].detach().numpy()
        s_categories = self.s_categories.detach().cpu().numpy()
        n = s.shape[0]
        pi_values = []
        for i in range(s_categories.shape[0]):
            index = []
            for j in range(n):
                if np.array_equal(s[j, :], s_categories[i, :]):
                    index.append(j)
            pi_values.append(
                policy_value(pi_hat[index, :], mu_1=nuisance_test["mu1"][index, :], mu_0=nuisance_test["mu0"][index, :],
                             prop=nuisance_test["prop"][index, :], a=data.data["a"][index, :], y=data.data["y"][index, :],
                             m=self.m, summary="mean"))

        for i in range(len(pi_values)):
            pi_values[i] = pi_values[i].detach().numpy()
        return pi_values

    def predict_ite(self, data):
        self.eval()
        if self.tarnet is not None:
            nuisance = self.tarnet.predict_nuisance(data.data)
            mu1 = nuisance["mu1"].detach().cpu().numpy()
            mu0 = nuisance["mu0"].detach().cpu().numpy()
            ite = mu1 - mu0
            return ite
        else:
            raise ValueError("Kein TARNet-Modell für ITE-Schätzung vorhanden.")
    
    def predict_cate(self, data, treat_cols, treat_values, base_values=None):
        """
        Schätzt den CATE für eine bestimmte Treatment-Kombination.
        treat_cols: Liste der Treatment-Spalten, z.B. ["trainy1", "trainy2"]
        treat_values: Werte für Treatment (z.B. [1, 1] für beide Trainingsjahre)
        base_values: Werte für Baseline (z.B. [0, 0] für kein Training)
        Gibt den Unterschied in der Outcome-Vorhersage zwischen treat_values und base_values zurück.
        """
        import copy
        # Kopien der Daten erzeugen
        data_treat = copy.deepcopy(data)
        data_base = copy.deepcopy(data)
        # Setze die Treatments auf die gewünschten Werte
        for col, val in zip(treat_cols, treat_values):
            idx = data_treat.data["x"].columns.get_loc(col) if hasattr(data_treat.data["x"], "columns") else col
            data_treat.data["x"][:, idx] = val
        if base_values is None:
            base_values = [0] * len(treat_cols)
        for col, val in zip(treat_cols, base_values):
            idx = data_base.data["x"].columns.get_loc(col) if hasattr(data_base.data["x"], "columns") else col
            data_base.data["x"][:, idx] = val
        # Potenzialwerte berechnen
        self.eval()
        nuisance_treat = self.tarnet.predict_nuisance(data_treat.data)
        nuisance_base = self.tarnet.predict_nuisance(data_base.data)
        mu_treat = nuisance_treat["mu1"].detach().cpu().numpy()
        mu_base = nuisance_base["mu0"].detach().cpu().numpy()
        cate = mu_treat - mu_base
        return cate
