import random
import os
import re
import numpy as np
import torch
import torch.utils.data
import librosa

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict
from yin import compute_yin


class TextMelLoader(torch.utils.data.Dataset):
    """
        **EDITED**by ju
        오디오 파일 경로와 텍스트만 받도록 수정
        1) loads audio, text
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms and f0s from audio file.
    """
    def __init__(self, audiopath, text, hparams,speaker_id=0):
        self.speaker_id = speaker_id
        self.audiopath = audiopath
        self.text = text
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.f0_min = hparams.f0_min
        self.f0_max = hparams.f0_max
        self.harm_thresh = hparams.harm_thresh
        self.p_arpabet = hparams.p_arpabet

        self.cmudict = None
        if hparams.cmudict_path is not None:
            self.cmudict = cmudict.CMUDict(hparams.cmudict_path)


    def get_f0(self, audio, sampling_rate=22050, frame_length=1024,
               hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
        f0, harmonic_rates, argmins, times = compute_yin(
            audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
            harm_thresh)
        pad = int((frame_length / hop_length) / 2)
        f0 = [0.0] * pad + f0 + [0.0] * pad

        f0 = np.array(f0, dtype=np.float32)
        return f0

    def get_data(self):
        text = self.get_text(self.text)
        mel, f0 = self.get_mel_and_f0(self.audiopath)
        return (text, mel, self.speaker_id, f0)

    def get_mel_and_f0(self, filepath):
        audio, sampling_rate = load_wav_to_torch(filepath)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        f0 = self.get_f0(audio.cpu().numpy(), self.sampling_rate,
                         self.filter_length, self.hop_length, self.f0_min,
                         self.f0_max, self.harm_thresh)
        f0 = torch.from_numpy(f0)[None]
        f0 = f0[:, :melspec.size(1)]

        return melspec, f0

    def get_text(self, text):
        text_norm = torch.IntTensor(
            text_to_sequence(text, self.text_cleaners, self.cmudict, self.p_arpabet))

        return text_norm


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        f0_padded = torch.FloatTensor(len(batch), 1, max_target_len)
        f0_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]
            f0 = batch[ids_sorted_decreasing[i]][3]
            f0_padded[i, :, :f0.size(1)] = f0

        model_inputs = (text_padded, input_lengths, mel_padded, gate_padded,
                        output_lengths, speaker_ids, f0_padded)

        return model_inputs
