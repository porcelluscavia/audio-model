import random
import numpy as np
import torch
import os
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import numpy as np
import os


# import tensorflow as tf
# tf.compat.v1.enable_eager_execution()



# import random
# import numpy as np
# import torch
# import os
#
#
# def get_start_end_idx(audio_size, clip_size, clip_idx, num_clips):
#     """
#     Sample a clip of size clip_size from an audio of size audio_size and
#     return the indices of the first and last sample of the clip. If clip_idx is
#     -1, the clip is randomly sampled, otherwise uniformly split the audio to
#     num_clips clips, and select the start and end index of clip_idx-th audio
#     clip.
#     Args:
#         audio_size (int): number of overall samples.
#         clip_size (int): size of the clip to sample from the samples.
#         clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
#             clip_idx is larger than -1, uniformly split the audio to num_clips
#             clips, and select the start and end index of the clip_idx-th audio
#             clip.
#         num_clips (int): overall number of clips to uniformly sample from the
#             given audio for testing.
#     Returns:
#         start_idx (int): the start sample index.
#         end_idx (int): the end sample index.
#     """
#     delta = max(audio_size - clip_size, 0)
#     if clip_idx == -1:
#         # Random temporal sampling.
#         start_idx = random.uniform(0, delta)
#     else:
#         # Uniformly sample the clip with the given index.
#         start_idx = np.linspace(0, delta, num=num_clips)[clip_idx]
#     end_idx = start_idx + clip_size - 1
#     return start_idx, end_idx
#
#
# def pack_audio(cfg, audio_record, temporal_sample_index):
#     path_audio = os.path.join(cfg.VGGSOUND.AUDIO_DATA_DIR, audio_record['video'][:-4] + '.wav')
#     import librosa
#     samples, sr = librosa.core.load(path_audio, sr=None, mono=False)
#     assert sr == cfg.AUDIO_DATA.SAMPLING_RATE, \
#         "Audio sampling rate ({}) does not match target sampling rate ({})".format(sr, cfg.AUDIO_DATA.SAMPLING_RATE)
#     start_idx, end_idx = get_start_end_idx(
#         samples.shape[0],
#         int(round(cfg.AUDIO_DATA.SAMPLING_RATE * cfg.AUDIO_DATA.CLIP_SECS)),
#         temporal_sample_index,
#         cfg.TEST.NUM_ENSEMBLE_VIEWS
#     )
#     spectrogram = _extract_sound_feature(cfg, samples, int(start_idx), int(end_idx))
#     return spectrogram
#
#
# def _log_specgram(cfg, audio, window_size=10,
#                  step_size=5, eps=1e-6):
#     nperseg = int(round(window_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
#     noverlap = int(round(step_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
#     from librosa import stft, filters
#
#     # mel-spec
#     spec = stft(audio, n_fft=2048,
#                 window='hann',
#                 hop_length=noverlap,
#                 win_length=nperseg,
#                 pad_mode='constant')
#     mel_basis = filters.mel(sr=cfg.AUDIO_DATA.SAMPLING_RATE,
#                             n_fft=2048,
#                             n_mels=128,
#                             htk=True,
#                             norm=None)
#     mel_spec = np.dot(mel_basis, np.abs(spec))
#
#     # log-mel-spec
#     log_mel_spec = np.log(mel_spec + eps)
#     return log_mel_spec.T
#
# #
# def _extract_sound_feature(cfg, samples, start_idx, end_idx):
#     if samples.shape[0] < int(round(cfg.AUDIO_DATA.SAMPLING_RATE * cfg.AUDIO_DATA.CLIP_SECS)):
#         spectrogram = _log_specgram(cfg, samples,
#                                     window_size=cfg.AUDIO_DATA.WINDOW_LENGTH,
#                                     step_size=cfg.AUDIO_DATA.HOP_LENGTH
#                                     )
#         num_timesteps_to_pad = cfg.AUDIO_DATA.NUM_FRAMES - spectrogram.shape[0]
#         spectrogram = np.pad(spectrogram, ((0, num_timesteps_to_pad), (0, 0)), 'edge')
#     else:
#         samples = samples[start_idx:end_idx]
#         spectrogram = _log_specgram(cfg, samples,
#                                     window_size=cfg.AUDIO_DATA.WINDOW_LENGTH,
#                                     step_size=cfg.AUDIO_DATA.HOP_LENGTH
#                                     )
#     return torch.tensor(spectrogram).unsqueeze(0)




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


def pack_audio(cfg, audio_record, temporal_sample_index):
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
    spectrogram = _extract_sound_feature(cfg, samples, int(start_idx), int(end_idx), has_torch=cfg.VGGSOUND.TORCH_STFT_INPUTS)
    # print(spectrogram)

    return spectrogram


def _log_specgram(cfg, audio, window_size=10,
                 step_size=5, eps=1e-6):
    nperseg = int(round(window_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
    noverlap = int(round(step_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
    from librosa import stft, filters

    # mel-spec
    spec = stft(audio, n_fft=2048,
                window='hann',
                hop_length=noverlap,
                win_length=nperseg,
                pad_mode='constant')
    # print(spec.shape)

    #what is the dimension of spec?

    mel_basis = filters.mel(sr=cfg.AUDIO_DATA.SAMPLING_RATE,
                            n_fft=2048,
                            n_mels=128,
                            htk=True,
                            norm=None)
    mel_spec = np.dot(mel_basis, np.abs(spec))

    # log-mel-spec
    log_mel_spec = np.log(mel_spec + eps)
    # print(log_mel_spec.T.shape)
    return log_mel_spec.T

def _log_specgram_torch(cfg, waveform):
    audio = torch.from_numpy(waveform)
    #audio = audio.cuda()
    windowsize = 2048
    # print(windowsize)
    # nperseg = int(round(window_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
    # noverlap = int(round(step_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
    window = torch.hann_window(windowsize)
    #window = window.cuda()
    from librosa import stft, filters
    filter_banks = filters.mel(sr=cfg.AUDIO_DATA.SAMPLING_RATE,
                            n_fft=2048,
                            n_mels=128,
                            htk=True,
                            norm=None)
    filter_banks = torch.from_numpy(filter_banks)
    #filter_banks = filter_banks.cuda()
    # S = torch.stft(audio, 2048,
    #             hop_length=noverlap,
    #             win_length=nperseg, window=window,
    #             pad_mode='constant').pow(2).sum(2).sqrt()
    S = torch.stft(audio, windowsize, window=window).pow(2).sum(2).sqrt()
    mel_S = filter_banks @ S
    log_mel_S = torch.log1p(mel_S)
    log_mel_S_T = torch.transpose(log_mel_S, 0, 1)
    # print(log_mel_S_T.shape)
    log_mel_S_T = log_mel_S_T.cpu()
    return log_mel_S_T

def _extract_sound_feature(cfg, samples, start_idx, end_idx, has_torch=False):

    if samples.shape[0] < int(round(cfg.AUDIO_DATA.SAMPLING_RATE * cfg.AUDIO_DATA.CLIP_SECS)):
        if has_torch:
            # This branch never seems to be implemented... *no error is thrown when there should be.)
            spectrogram = _log_specgram_torch(cfg, samples)
        else:
            spectrogram = _log_specgram(cfg, samples,
                                    window_size=cfg.AUDIO_DATA.WINDOW_LENGTH,
                                    step_size=cfg.AUDIO_DATA.HOP_LENGTH
                                    )

        num_timesteps_to_pad = cfg.AUDIO_DATA.NUM_FRAMES - spectrogram.shape[0]
        spectrogram = np.pad(spectrogram, ((0, num_timesteps_to_pad), (0, 0)), 'edge')
        spectrogram = torch.tensor(spectrogram)
    else:
        samples = samples[start_idx:end_idx]
        if has_torch:
            spectrogram = _log_specgram_torch(cfg, samples)
            spectrogram = spectrogram.cpu().detach().numpy()

            # Currently: spectrogram.shape[0] is 160, num_timesteps_to_pad is 352
            num_timesteps_to_pad = cfg.AUDIO_DATA.NUM_FRAMES - spectrogram.shape[0]
            #print(spectrogram.shape)
            # Concatenate num_timesteps_to_pad rows of the final row to our spectrogram- essentially stretch out the last bit
            spectrogram = np.pad(spectrogram, ((0, num_timesteps_to_pad), (0, 0)), 'edge')
            # Normalize values.
            scaler = MinMaxScaler(feature_range=(-10, 10))
            spectrogram = scaler.fit_transform(spectrogram)


        else:
            spectrogram = _log_specgram(cfg, samples,
                                    window_size=cfg.AUDIO_DATA.WINDOW_LENGTH,
                                    step_size=cfg.AUDIO_DATA.HOP_LENGTH
                           )
        # spec_file = os.path.join(cfg.VGGSOUND.FILE_DIR, 'saved_specs.npy')
        # p = Path(spec_file)
        # with p.open('ab') as f:
        #     np.save(f, spectrogram)
        spectrogram = torch.tensor(spectrogram)
    #print(torch.count_nonzero(spectrogram))

    return spectrogram.unsqueeze(0)

# def _extract_sound_feature(cfg, samples, start_idx, end_idx):
#     if samples.shape[0] < int(round(cfg.AUDIO_DATA.SAMPLING_RATE * cfg.AUDIO_DATA.CLIP_SECS)):
#         spectrogram = _log_specgram(cfg, samples,
#                                     window_size=cfg.AUDIO_DATA.WINDOW_LENGTH,
#                                     step_size=cfg.AUDIO_DATA.HOP_LENGTH
#                                     )
#         num_timesteps_to_pad = cfg.AUDIO_DATA.NUM_FRAMES - spectrogram.shape[0]
#         spectrogram = np.pad(spectrogram, ((0, num_timesteps_to_pad), (0, 0)), 'edge')
#     else:
#         samples = samples[start_idx:end_idx]
#         spectrogram = _log_specgram(cfg, samples,
#                                     window_size=cfg.AUDIO_DATA.WINDOW_LENGTH,
#                                     step_size=cfg.AUDIO_DATA.HOP_LENGTH
#                                     )
#     return torch.tensor(spectrogram).unsqueeze(0)


def recover_audio(cfg, audio_tensor, eps=1e-6):
    """
    Reconstruct an audio file based on a spectrogram that has been numerically manipulated.
    It should come in with shape torch.Size([1, 1, 3, 128, 128].
       Args:
           audio_tensor (torch Tensor): the manipulated spectrogram Tensor
           eps (float): value added to original spectrogram to avoid division by zero

       Returns:
           recovered_audio (torch Tensor): the start sample index.
       """

    from librosa import feature
    #feed the untouched spectrogram in instead? but gradcam makes filters on the mel spectrogram

    # This audio is a spectrogram with mel filter applied to it.
    audio_tensor = torch.squeeze(audio_tensor)

    audio_tensor = audio_tensor.T

    log_mel_array = audio_tensor.cpu().detach().numpy()
    # import pdb
    # pdb.set_trace()
    # All 3 arrays returned are identical.
    mel_array = np.exp(log_mel_array) - eps
    # copy the parameters from forward step. Else, look up how to do inversion for steps separately.
    recovered_audio = feature.inverse.mel_to_audio(mel_array[0], sr= cfg.AUDIO_DATA.SAMPLING_RATE)



    #TODO later later: improve the audio with NN's?
    return recovered_audio




