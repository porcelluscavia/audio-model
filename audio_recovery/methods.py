#Inspired by the (non-working) torch_specinv python package (https://github.com/yoyololicon/spectrogram-inversion/blob/master/torch_specinv/methods.py)

from .metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lbfgs import LBFGS
from tqdm import tqdm

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

    audio = audio.cuda()
    window = window.cuda()
    nperseg = int(round(window_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
    noverlap = int(round(step_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
    window = torch.hann_window(nperseg)
    from librosa import stft, filters
    filter_banks = filters.mel(sr=cfg.AUDIO_DATA.SAMPLING_RATE,
                            n_fft=2048,
                            n_mels=128,
                            htk=True,
                            norm=None)
    filter_banks = torch.from_numpy(filter_banks).cuda()
    filter_banks = torch.from_numpy(filter_banks)
    S = torch.stft(audio, 2048,
                hop_length=noverlap,
                win_length=nperseg, window=window,
                pad_mode='constant').pow(2).sum(2).sqrt()
    mel_S = filter_banks @ S
    log_mel_S = torch.log1p(mel_S)
    log_mel_S_T = torch.transpose(log_mel_S, 0, 1)
    #np_log_mel_S = log_mel_S_T.cpu().detach().numpy()
    return log_mel_S_T



def iteratively_recover_audio(cfg, file_name, temporal_sample_index):
    waveform = pack_audio(cfg, file_name, temporal_sample_index)
    waveform_tensor = torch.from_numpy(waveform)
    #y = torch.from_numpy(waveform_tensor)
    mag = trsfn(waveform_tensor)

    yhat = L_BFGS(mag, trsfn, len(waveform_tensor))
    yhat_cpu = yhat.cpu()
    waveform_tensor_cpu = waveform_tensor.cpu()

    return yhat_cpu, waveform_tensor_cpu





def get_start_end_idx(audio_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from an audio of size audio_size and
    return the indices of the first and last sample of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the audio to
    num_clips clips, and select the start and end index of clip_idx-th audio
    clip.
    Args:
        audio_size (int): number of overall samples.
        clip_size (int): size of the clip to sample from the samples.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the audio to num_clips
            clips, and select the start and end index of the clip_idx-th audio
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given audio for testing.
    Returns:
        start_idx (int): the start sample index.
        end_idx (int): the end sample index.
    """
    delta = max(audio_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = np.linspace(0, delta, num=num_clips)[clip_idx]
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def pack_audio(cfg, audio_record, temporal_sample_index, torch = True):
    path_audio = os.path.join(cfg.VGGSOUND.AUDIO_DATA_DIR, audio_record['video'][:-4] + '.wav')
    import librosa
    samples, sr = librosa.core.load(path_audio, sr=None, mono=False)
    assert sr == cfg.AUDIO_DATA.SAMPLING_RATE, \
        "Audio sampling rate ({}) does not match target sampling rate ({})".format(sr, cfg.AUDIO_DATA.SAMPLING_RATE)
    start_idx, end_idx = get_start_end_idx(
        samples.shape[0],
        int(round(cfg.AUDIO_DATA.SAMPLING_RATE * cfg.AUDIO_DATA.CLIP_SECS)),
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS
    )
    clipped_audio = samples[start_idx:end_idx]
    return clipped_audio

