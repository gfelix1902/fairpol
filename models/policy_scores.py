import torch
from torch import Tensor


def score_dm(pi: Tensor, mu_1: Tensor, mu_0: Tensor) -> Tensor:
    return pi * mu_1 + (1 - pi) * mu_0


def score_ipw(pi, prop, a, y: Tensor) -> Tensor:
    nom = a * pi + (1 - a) * (1 - pi)
    denom = a * prop + (1 - a) * (1 - prop)
    return (nom / denom) * y


def score_dr(pi, mu_1, mu_0, prop, a, y: Tensor) -> Tensor:
    pi_a = a * pi + (1 - a) * (1 - pi)
    e_a = a * prop + (1 - a) * (1 - prop)
    mu_a = a * mu_1 + (1 - a) * mu_0
    return score_dm(pi, mu_1, mu_0) + pi_a * ((y - mu_a) / e_a)


def score_m(pi, mu_1=None, mu_0=None, prop=None, a=None, y=None, m="dm"):
    if m == "dm":
        if None not in [pi, mu_1, mu_0]:
            return score_dm(pi, mu_1, mu_0)
    if m == "ipw":
        if None not in [pi, prop, a, y]:
            return score_ipw(pi, prop, a, y)
    if m == "dr":
        if None not in [pi, mu_1, mu_0, prop, a, y]:
            return score_dr(pi, mu_1, mu_0, prop, a, y)


def policy_value(pi, mu_1=None, mu_0=None, prop=None, a=None, y=None, m="dm", summary="mean") -> Tensor:
    if summary == "mean":
        return torch.mean(score_m(pi, mu_1, mu_0, prop, a, y, m))
    elif summary == "sum":
        return torch.sum(score_m(pi, mu_1, mu_0, prop, a, y, m))
