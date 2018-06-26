from __future__ import division

import math
import numpy as np

import torch
from torch.autograd import Variable


def batch_slices(n_samples, batch_size=32):
    n_batches = math.ceil(n_samples / batch_size)
    batches = [slice(ix * batch_size, (ix + 1) * batch_size)
               for ix in range(n_batches)]
    return batches


def S_from_Ainv(Ainv):
    """See footnote in notes.pdf"""

    # Ainv = torch.FloatTensor(Ainv).view(1 + n_active, 1 + n_active)
    S = Ainv[1:, 1:]
    k = Ainv[0, 0]
    b = Ainv[0, 1:].unsqueeze(0)

    S -= (1 / k) * (b * b.t())
    return S


def expand_with_zeros(x, rows, cols):
    orig_rows, orig_cols = x.size()

    ret = x
    if orig_cols < cols:
        horiz = x.new_zeros((orig_rows, cols - orig_cols))
        ret = torch.cat([ret, horiz], dim=-1)

    if orig_rows < rows:
        vert = x.new_zeros((rows - orig_rows, cols))
        ret = torch.cat([ret, vert], dim=0)

    return ret
