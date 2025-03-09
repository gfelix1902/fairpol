from abc import ABC, abstractmethod

import numpy
import numpy as np
import pytorch_lightning as pl
import models.policy_scores as ps
import torch.nn as nn
from scipy.stats import wasserstein_distance


class PolicyNet(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()

    # Input: dataset, Output: Policy predictions
    @abstractmethod
    def predict(self, data):
        pass

    def log_dict_prefix(self, loss_dict, prefix=""):
        # Rename keys with prefix
        loss_dict_prefix = dict([(prefix + "_" + key, value) for key, value in loss_dict.items()])
        self.log_dict(loss_dict_prefix, logger=True, on_epoch=True, on_step=False)

    # Evaluate learned policy on test data
    def evaluate_policy(self, data_test, oracle=True, nuisance=None, m="dm"):
        pi_hat = self.predict(data_test)
        if oracle:
            policy_value = ps.policy_value(pi=pi_hat, mu_1=data_test.nuisance["mu1"], mu_0=data_test.nuisance["mu0"],
                                           prop=data_test.nuisance["prop"], m=m)
        else:
            policy_value = ps.policy_value(pi=pi_hat, mu_1=nuisance["mu1"], mu_0=nuisance["mu0"], prop=nuisance["prop"],
                                           m=m, a=data_test.data["a"], y=data_test.data["y"])
        return policy_value.detach().numpy()

    def evaluate_policy_perturbed(self, data_test_perturbed):
        p_values = []
        for d_test in data_test_perturbed:
            p_values.append(self.evaluate_policy(d_test))
        return p_values

    def evaluate_worst_case(self, data_test_perturbed):
        return min(self.evaluate_policy_perturbed(data_test_perturbed))

    def evaluate_action_fairness(self, data_test_perturbed):
        #Check action fairness, evaluate on perturbed dataset and check differences in prediction distributions
        predictions0 = self.predict(data_test_perturbed[1]).detach().numpy()
        predictions1 = self.predict(data_test_perturbed[2]).detach().numpy()
        #Wasserstein distance between prediction distributions
        #dist = wasserstein_distance(u_values=predictions0[:, 0], v_values=predictions1[:, 0])
        dist = np.mean(np.abs(predictions1 - predictions0))
        return dist

    def evaluate_action_fairness_repr(self, data_test_perturbed):
        #Check action fairness, evaluate on perturbed dataset and check differences in prediction distributions
        if hasattr(self, "repr_net"):
            if self.repr_net is not None:
                repr0 = self.repr_net.predict_repr(data_test_perturbed[1].data).detach().numpy()
                repr1 = self.repr_net.predict_repr(data_test_perturbed[2].data).detach().numpy()
                #Wasserstein distance between prediction distributions
                #dist = wasserstein_distance(u_values=predictions0[:, 0], v_values=predictions1[:, 0])
                dist = np.mean(np.sqrt(np.sum((repr0 - repr1)**2, axis=1)))
                return dist
            else:
                return numpy.NAN
        else:
            return numpy.NAN
