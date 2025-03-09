import torch
from torch.utils.data import Dataset
import numpy as np


class Static_Dataset(Dataset):
    def __init__(self, y, a, x, s, y_type, x_type, s_type, dim_xu=None, mu1=None, mu0=None, prop=None, mu1_f=None, mu0_f=None,
                 mu1_s1=None, mu1_s0=None):
        self.scaling_params = {"y": {"m": 0, "sd": 1}, "x": {"m": 0, "sd": 1}, "s": {"m": 0, "sd": 1}}
        self.data = {"y": y, "a": a, "x": x, "s": s}
        self.datatypes = {"y_type": y_type, "x_type": x_type, "s_type": s_type}
        self.dim_xu = dim_xu
        self.nuisance = {"mu1": mu1, "mu0": mu0, "prop": prop, "mu1_f": mu1_f, "mu0_f": mu0_f, "mu1_s1": mu1_s1, "mu1_s0": mu1_s0}
        self.load_nuisance = False

    def __len__(self):
        return self.data["y"].shape[0]

    def set_load_nuisance(self, load_nuisance):
        self.load_nuisance = load_nuisance

    def __getitem__(self, index) -> dict:
        if not self.load_nuisance:
            return {k: v[index] for k, v in self.data.items()}
        else:
            return {**{k: v[index] for k, v in self.data.items()}, **{k: v[index] for k, v in self.nuisance.items()}}

    # Scaling------------------
    def scale_y(self, log_transform=False):
        if self.datatypes["y_type"] == "continuous":
            if log_transform:
                self.data["y"] = np.log(self.data["y"])
            m = np.squeeze(np.mean(self.data["y"], axis=0))
            sd = np.squeeze(np.std(self.data["y"], axis=0))
            self.data["y"] = self.__scale_vector(self.data["y"], m, sd)
            self.scaling_params["y"]["m"] = m
            self.scaling_params["y"]["sd"] = sd

    def unscale_y(self):
        if self.datatypes["y_type"] == "continuous":
            m = self.scaling_params["y"]["m"]
            sd = self.scaling_params["y"]["sd"]
            self.data["y"] = self.__unscale_vector(self.data["y"], m, sd)
            self.scaling_params["y"]["m"] = 0
            self.scaling_params["y"]["sd"] = 1

    def scale_cov(self):
        m_x = np.squeeze(np.mean(self.data["x"], axis=0))
        sd_x = np.squeeze(np.std(self.data["x"], axis=0))
        m_s = np.squeeze(np.mean(self.data["s"], axis=0))
        sd_s = np.squeeze(np.std(self.data["s"], axis=0))
        for i, type in enumerate(self.datatypes["x_type"]):
            if type == "continuous":
                self.data["x"][:, i] = self.__scale_vector(self.data["x"][:, i], m_x[i], sd_x[i])
        self.data["s"][:, self.datatypes["s_type"] == "continuous"] = self.__scale_vector(
            self.data["s"][:, self.datatypes["s_type"] == "continuous"], m_s, sd_s)
        self.scaling_params["x"]["m"] = m_x
        self.scaling_params["x"]["sd"] = sd_x
        self.scaling_params["s"]["m"] = m_s
        self.scaling_params["s"]["sd"] = sd_s

    def unscale_cov(self):
        m_x = self.scaling_params["x"]["m"]
        sd_x = self.scaling_params["x"]["sd"]
        m_s = self.scaling_params["s"]["m"]
        sd_s = self.scaling_params["s"]["sd"]
        self.data["x"][:, self.datatypes["x_type"] == "continuous"] = self.__unscale_vector(
            self.data["x"][:, self.datatypes["x_type"] == "continuous"], m_x, sd_x)
        self.data["s"][:, self.datatypes["s_type"] == "continuous"] = self.__unscale_vector(
            self.data["s"][:, self.datatypes["s_type"] == "continuous"], m_s, sd_s)

    def standardize(self, log_transform_y=False):
        self.scale_y(log_transform_y)
        self.scale_cov()

    @staticmethod
    def __scale_vector(data, m, sd):
        return (data - m) / sd

    @staticmethod
    def __unscale_vector(data, m, sd):
        return (data * sd) + m

    def convert_to_tensor(self):
        self.data["y"] = torch.from_numpy(self.data["y"].astype(np.float32))
        self.data["a"] = torch.from_numpy(self.data["a"].astype(np.float32))
        self.data["x"] = torch.from_numpy(self.data["x"].astype(np.float32))
        self.data["s"] = torch.from_numpy(self.data["s"].astype(np.float32))
        if self.nuisance["mu1"] is not None:
            self.nuisance["mu1"] = torch.from_numpy(self.nuisance["mu1"].astype(np.float32))
        if self.nuisance["mu0"] is not None:
            self.nuisance["mu0"] = torch.from_numpy(self.nuisance["mu0"].astype(np.float32))
        if self.nuisance["prop"] is not None:
            self.nuisance["prop"] = torch.from_numpy(self.nuisance["prop"].astype(np.float32))
        if self.nuisance["mu1_f"] is not None:
            self.nuisance["mu1_f"] = torch.from_numpy(self.nuisance["mu1_f"].astype(np.float32))
        if self.nuisance["mu0_f"] is not None:
            self.nuisance["mu0_f"] = torch.from_numpy(self.nuisance["mu0_f"].astype(np.float32))
        if self.nuisance["mu1_s1"] is not None:
            self.nuisance["mu1_s1"] = torch.from_numpy(self.nuisance["mu1_s1"].astype(np.float32))
        if self.nuisance["mu1_s0"] is not None:
            self.nuisance["mu1_s0"] = torch.from_numpy(self.nuisance["mu1_s0"].astype(np.float32))

    def convert_to_np(self):
        self.data["y"] = self.data["y"].detach().numpy()
        self.data["a"] = self.data["a"].detach().numpy()
        self.data["x"] = self.data["x"].detach().numpy()
        self.data["s"] = self.data["s"].detach().numpy()
        self.nuisance["mu1"] = self.nuisance["mu1"].detach().numpy()
        self.nuisance["mu0"] = self.nuisance["mu0"].detach().numpy()
        self.nuisance["prop"] = self.nuisance["prop"].detach().numpy()
        self.nuisance["mu1_f"] = self.nuisance["mu1_f"].detach().numpy()
        self.nuisance["mu0_f"] = self.nuisance["mu0_f"].detach().numpy()
        self.nuisance["mu1_s1"] = self.nuisance["mu1_s1"].detach().numpy()
        self.nuisance["mu1_s0"] = self.nuisance["mu1_s0"].detach().numpy()

