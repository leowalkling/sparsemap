import torch
from torch.autograd import Function
from torch import nn
import numpy as np

from .. import sparsemap
from ..utils import S_from_Ainv


def _d_vbar(M, dy, Ainv):
    S = S_from_Ainv(Ainv)

    if M.is_cuda:
        S = S.cuda()
    # B = S11t / 1S1t
    # dvbar = (I - B) S M dy

    # we first compute S M dy
    first_term = S @ (M @ dy)
    # then, BSMt dy = B * first_term. Optimized:
    # 1S1t = S.sum()
    # S11tx = (S1) (1t * x)
    second_term = (first_term.sum() * S.sum(0)) / S.sum()
    d_vbar = first_term - second_term
    return d_vbar


class _BaseSparseMarginalsFunction(Function):
    @staticmethod
    def forward(ctx, factor, unaries, max_iter, verbose):
        cuda_device = None
        if unaries.is_cuda:
            cuda_device = unaries.get_device()
            unaries = unaries.cpu()

        original_shape = unaries.size()
        u, _, status = sparsemap(factor, unaries.contiguous().view(-1), [],
                                 max_iter=max_iter,
                                 verbose=verbose)

        ctx.sparsemap_M = torch.from_numpy(status["M"])
        ctx.sparsemap_inverse_A = torch.from_numpy(status["inverse_A"])

        out = torch.from_numpy(np.ascontiguousarray(u).reshape(*original_shape))
        if cuda_device is not None:
            out = out.cuda(cuda_device)
        return out

    @staticmethod
    def backward(ctx, dy):
        cuda_device = None

        if dy.is_cuda:
            cuda_device = dy.get_device()
            dy = dy.cpu()

        M = ctx.sparsemap_M
        Ainv = ctx.sparsemap_inverse_A
        # if cuda_device is not None:
        #     M = M.cuda()
        #     Ainv = Ainv.cuda()

        d_vbar = _d_vbar(M, dy.contiguous().view(-1), Ainv)
        d_unary = M.t() @ d_vbar

        if cuda_device is not None:
            d_unary = d_unary.cuda(cuda_device)

        return None, d_unary.contiguous().view_as(dy), None, None


_base_sparsemap_function = _BaseSparseMarginalsFunction.apply


class _BaseSparseMarginals(nn.Module):
    def __init__(self, max_iter=10, verbose=0):
        super(_BaseSparseMarginals, self).__init__()
        self.max_iter = max_iter
        self.verbose = verbose

    def build_factor(self, *sizes):
        raise NotImplementedError()

    def forward(self, unaries):
        factor = self.build_factor(*unaries.size())
        u = _base_sparsemap_function(factor, unaries, self.max_iter, self.verbose)
        return u


class _BaseSparseMarginalsFunctionAdditionals(Function):
    @staticmethod
    def forward(ctx, factor, unaries, additionals, max_iter, verbose):
        cuda_device = None
        if unaries.is_cuda:
            cuda_device = unaries.get_device()
            unaries = unaries.cpu()
            additionals = additionals.cpu()

        original_shape = unaries.size()
        u, uadd, status = sparsemap(factor, unaries.contiguous().view(-1), additionals,
                                    max_iter=max_iter,
                                    verbose=verbose)

        ctx.sparsemap_M = torch.from_numpy(status["M"])
        ctx.sparsemap_Madd = torch.from_numpy(status["Madd"])
        ctx.sparsemap_inverse_A = torch.from_numpy(status["inverse_A"])

        out = torch.from_numpy(np.ascontiguousarray(u).reshape(*original_shape))
        if cuda_device is not None:
            out = out.cuda(cuda_device)
        return out

    @staticmethod
    def backward(ctx, dy):
        cuda_device = None

        if dy.is_cuda:
            cuda_device = dy.get_device()
            dy = dy.cpu()

        M = ctx.sparsemap_M
        Madd = ctx.sparsemap_Madd
        Ainv = ctx.sparsemap_inverse_A
        if cuda_device is not None:
            M = M.cuda(cuda_device)
            Madd = Madd.cuda(cuda_device)
            Ainv = Ainv.cuda(cuda_device)

        d_vbar = _d_vbar(M, dy.contiguous().view(-1), Ainv)
        d_unary = M.t() @ d_vbar
        d_additionals = Madd.t() @ d_vbar

        if cuda_device is not None:
            d_unary = d_unary.cuda(cuda_device)
            d_additionals = d_additionals.cuda(cuda_device)

        return None, d_unary.contiguous().view_as(dy), d_additionals, None, None


_base_sparsemap_function_with_additionals = _BaseSparseMarginalsFunctionAdditionals.apply


class _BaseSparseMarginalsAdditionals(_BaseSparseMarginals):
    def forward(self, unaries, additionals):
        n_variables, n_states = unaries.size()
        factor = self.build_factor(n_variables, n_states)

        return _base_sparsemap_function_with_additionals(factor, unaries, additionals, self.max_iter, self.verbose)
