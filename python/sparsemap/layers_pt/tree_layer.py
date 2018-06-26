# pytorch layer that applies a TreeFactor

import numpy as np
import torch

from ad3 import PFactorGraph
from ad3.extensions import PFactorTree

from .base import _BaseSparseMarginals
from .._factors import PFactorTreeFast


class TreeSparseMarginals(_BaseSparseMarginals):
    def __init__(self, n_nodes=None, max_iter=10, verbose=0):
        super(TreeSparseMarginals, self).__init__(max_iter, verbose)
        self.n_nodes = n_nodes
        self.arcs = [(h, m)
                     for m in range(1, n_nodes + 1)
                     for h in range(n_nodes + 1)
                     if h != m]

    def build_factor(self, n_arcs):
        g = PFactorGraph()
        n_nodes = self.n_nodes
        arcs = self.arcs
        if n_arcs != len(arcs):
            raise ValueError("expected input dim of {:d} but got {:d}".format(len(arcs), n_arcs))
        arc_vars = [g.create_binary_variable() for _ in arcs]
        tree = PFactorTree()
        g.declare_factor(tree, arc_vars)
        tree.initialize(n_nodes + 1, arcs)
        return tree


class TreeSparseMarginalsFast(_BaseSparseMarginals):

    def __init__(self, max_iter=10, verbose=0):
        super(TreeSparseMarginalsFast, self).__init__(max_iter, verbose)
        self.n_nodes = n_nodes
        self.arcs = [(h, m)
                     for m in range(1, n_nodes + 1)
                     for h in range(n_nodes + 1)
                     if h != m]

    def build_factor(self, n_arcs):
        g = PFactorGraph()
        n_nodes = self.n_nodes
        arcs = self.arcs
        if n_arcs != len(arcs):
            raise ValueError("expected input dim of {:d} but got {:d}".format(len(arcs), n_arcs))
        arc_vars = [g.create_binary_variable() for _ in arcs]
        tree = PFactorTreeFast()
        g.declare_factor(tree, arc_vars)
        tree.initialize(n_nodes + 1)
        return tree


if __name__ == '__main__':
    n_nodes = 3
    Wt = torch.randn((n_nodes + 1) * n_nodes, requires_grad=True)

    Wskip_a = []
    k = 0
    for m in range(1, n_nodes + 1):
        for h in range(n_nodes + 1):
            if h != m:
                Wskip_a.append(Wt[k])
            k += 1

    Wskip = torch.tensor(Wskip_a, requires_grad=True)

    tsm_slow = TreeSparseMarginals(n_nodes)
    posteriors = tsm_slow(Wskip)
    print("posteriors slow", posteriors)

    W = torch.tensor(Wskip_a, requires_grad=True)
    tsm = TreeSparseMarginalsFast(n_nodes)
    posteriors = tsm(W)
    print("posteriors fast", posteriors)
    posteriors.sum().backward()
    print("dposteriors_dW", W.grad)
