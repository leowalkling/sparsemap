# Segmental Semi-markov layer

from ad3 import PFactorGraph

import torch
from torch.autograd import Function

from .base import _BaseSparseMarginals
from .._factors import PFactorMatching


class MatchingSparseMarginals(_BaseSparseMarginals):
    def build_factor(self, n_rows, n_cols):
        match = PFactorMatching()
        match.initialize(n_rows, n_cols)
        return match


if __name__ == '__main__':

    n_rows = 5
    n_cols = 3
    scores = torch.randn(n_rows, n_cols, requires_grad=True)

    matcher = MatchingSparseMarginals()
    matching = matcher(scores)

    print(matching)
    matching.sum().backward()

    print("dpost_dunary", scores.grad)
