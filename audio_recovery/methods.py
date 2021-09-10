#Inspired by the (non-working) torch_specinv python package (https://github.com/yoyololicon/spectrogram-inversion/blob/master/torch_specinv/methods.py)

from .metrics import *
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.optim.lbfgs import LBFGS
from tqdm import tqdm
from functools import partial
from typing import Tuple
import math



_func_mapper = {
    'SC': sc,
    'SNR': snr,
    'SER': ser
}

def L_BFGS(spec, transform_fn, samples=None, init_x0=None, max_iter=1000, tol=1e-6, verbose=1, eva_iter=10, metric='sc',
           **kwargs):
    r"""

    Reconstruct spectrogram phase using `Inversion of Auditory Spectrograms, Traditional Spectrograms, and Other
    Envelope Representations`_, where I directly use the :class:`torch.optim.LBFGS` optimizer provided in PyTorch.
    This method doesn't restrict to traditional short-time Fourier Transform, but any kinds of presentation (ex: Mel-scaled Spectrogram) as
    long as the transform function is differentiable.

    .. _`Inversion of Auditory Spectrograms, Traditional Spectrograms, and Other Envelope Representations`:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6949659

    Args:
        spec (Tensor): the input presentation.
        transform_fn: a function that has the form ``spec = transform_fn(x)`` where x is an 1d tensor.
        samples (int, optional): number of samples in time domain. Default: :obj:`None`
        init_x0 (Tensor, optional): an 1d tensor that make use as initial time domain samples. If not provided, will use random
            value tensor with length equal to ``samples``.
        max_iter (int): maximum number of iterations before timing out.
        tol (float): tolerance of the stopping condition base on L2 loss. Default: ``1e-6``.
        verbose (bool): whether to be verbose. Default: :obj:`True`
        eva_iter (int): steps size for evaluation. After each step, the function defined in ``metric`` will evaluate. Default: ``10``
        metric (str): evaluation function. Currently available functions: ``'sc'`` (spectral convergence), ``'snr'`` or ``'ser'``. Default: ``'sc'``
        **kwargs: other arguments that pass to :class:`torch.optim.LBFGS`.

    Returns:
        A 1d tensor converted from the given presentation
    """
    running_loss = 0.0
    if init_x0 is None:
        init_x0 = spec.new_empty(*[samples]).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec
    T = torch.nn.functional.normalize(T)


    criterion = nn.MSELoss()
    optimizer = LBFGS([x], **kwargs)

    def inner_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        V = transform_fn(x)
        V = torch.nn.functional.normalize(V)
        loss = criterion(V, T)
        if loss.requires_grad:
            loss.backward()
        return loss

    def outer_closure():
        optimizer.step(inner_closure)
        with torch.no_grad():
            V = transform_fn(x)
        return V



    _training_loop_bfg(
        outer_closure,
        inner_closure,
        T,
        max_iter,
        tol,
        verbose,
        eva_iter,
        metric
    )

    return x.detach()


def _training_loop_bfg(
        outer_closure,
        inner_closure,
        target,
        max_iter,
        tol,
        verbose,
        eva_iter,
        metric,
):
    assert eva_iter > 0
    assert max_iter > 0
    assert tol >= 0

    metric = metric.upper()
    assert metric.upper() in _func_mapper.keys()

    bar_dict = {}
    bar_dict[metric] = 0
    metric_func = _func_mapper[metric]

    criterion = F.mse_loss
    init_loss = None

    with tqdm(total=max_iter, disable=not verbose) as pbar:
        for i in range(max_iter):

            output = outer_closure()
            # print(output)
            if i % eva_iter == eva_iter - 1:
                bar_dict[metric] = metric_func(output, target).item()
                loss = inner_closure()
                pbar.set_postfix(**bar_dict, loss=inner_closure())
                pbar.update(eva_iter)

                if not init_loss:
                    init_loss = loss
                elif (previous_loss - loss) / init_loss < tol and previous_loss > loss:
                    break
                previous_loss = loss


y = torch.from_numpy(waveform)
windowsize = 2048
window = torch.hann_window(windowsize)

filter_banks = torch.from_numpy(filters.mel(SAMPLING_RATE, windowsize)).cuda()
window = window.cuda()


def trsfn(x):
    S = torch.stft(x, windowsize, window=window).pow(2).sum(2).sqrt()
    mel_S = filter_banks @ S
    return torch.log1p(mel_S)


y = y.cuda()
mag = trsfn(y)

yhat = L_BFGS(mag, trsfn, len(y))

