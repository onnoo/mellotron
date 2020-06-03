import matplotlib
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append('waveglow/')

from itertools import cycle
import numpy as np
import scipy as sp
from scipy.io.wavfile import write
import pandas as pd
import librosa
import torch

from hparams import create_hparams
from model import Tacotron2, load_model
from waveglow.denoiser import Denoiser
from layers import TacotronSTFT
from inference_utils import TextMelLoader, TextMelCollate
from text import cmudict, text_to_sequence

class LoadedMellotron:
    def __init__(self, ckpt, wglw, n_speakers=123):
        print("[Loading Model]")
        self.ckpt = ckpt
        self.hparams = create_hparams()
        self.hparams.n_speakers = n_speakers
        self.stft = TacotronSTFT(self.hparams.filter_length, self.hparams.hop_length, self.hparams.win_length,
                            self.hparams.n_mel_channels, self.hparams.sampling_rate, self.hparams.mel_fmin,
                            self.hparams.mel_fmax)
        self.mellotron = load_model(self.hparams).cuda().eval()
        self.waveglow = torch.load(wglw)['model'].cuda().eval()
        self.denoiser = Denoiser(self.waveglow).cuda().eval()
        self.arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')   
        self.mellotron.load_state_dict(torch.load(ckpt)['state_dict'])
        print('[Loaded Model]')
    
    def load_mel(self, path):
        audio, sampling_rate = librosa.core.load(path, sr=self.hparams.sampling_rate)
        audio = torch.from_numpy(audio)
        if sampling_rate != self.hparams.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.hparams.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = melspec.cuda()
        return melspec
    
    def run(self, audio_path, text,title, speaker_id=0, ):
        print("[Running]")
        dataloader = TextMelLoader(audio_path, text, self.hparams, speaker_id)
        datacollate = TextMelCollate(1)

        text_encoded = torch.LongTensor(text_to_sequence(text, self.hparams.text_cleaners, self.arpabet_dict))[None, :].cuda()    
        pitch_contour = dataloader.get_data()[3][None].cuda()
        mel = self.load_mel(audio_path)
        print(audio_path, text)

        # load source data to obtain rhythm using tacotron 2 as a forced aligner
        x, y = self.mellotron.parse_batch(datacollate([dataloader.get_data()]))

        with torch.no_grad():
            # get rhythm (alignment map) using tacotron 2
            mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = self.mellotron.forward(x)
            rhythm = rhythm.permute(1, 0, 2)
        
        s_id = torch.LongTensor([speaker_id]).cuda()
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, _ = self.mellotron.inference_noattention(
                (text_encoded, mel, s_id, pitch_contour, rhythm))
            audio = self.denoiser(self.waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.02)[:, 0]  
        # plot_mel_f0_alignment(x[2].data.cpu().numpy()[0],
        #                 mel_outputs_postnet.data.cpu().numpy()[0],
        #                 pitch_contour.data.cpu().numpy()[0, 0],
        #                 rhythm.data.cpu().numpy()[:, 0].T, f"tests/{title}.png")
        write(f"outputs/{title}", rate=self.hparams.sampling_rate, data=audio[0].data.cpu().numpy())
        print("[END]")


    

def plot_mel_f0_alignment(mel_source, mel_outputs_postnet, f0s, alignments, title, figsize=(16, 16)):
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].imshow(mel_source, aspect='auto', origin='bottom', interpolation='none')
    axes[1].imshow(mel_outputs_postnet, aspect='auto', origin='bottom', interpolation='none')
    axes[2].scatter(range(len(f0s)), f0s, alpha=0.5, color='red', marker='.', s=1)
    axes[2].set_xlim(0, len(f0s))
    axes[3].imshow(alignments, aspect='auto', origin='bottom', interpolation='none')
    axes[0].set_title("Source Mel")
    axes[1].set_title("Predicted Mel")
    axes[2].set_title("Source pitch contour")
    axes[3].set_title("Source rhythm")
    plt.tight_layout()
    plt.savefig(title, dpi=300, bbox_inches='tight')
    plt.close(fig)
 

if __name__ == "__main__":
    mo = LoadedMellotron("outdir_kc8/checkpoint_65000", '../models/waveglow_256channels_v4.pt')
    mo.run("/home/jwyang/dataset/test_set/test_female.wav","누군가 말했죠 전체는 부분의 합보다 더 큰 법이라고요")
    