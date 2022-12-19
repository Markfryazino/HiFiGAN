import torch
import torch.nn.functional as F
import random
import torchaudio
import os

from src.config import TrainingConfig, MelSpectrogramConfig
from src.mel import MelSpectrogram


class MelWavDataset(torch.utils.data.Dataset):
    def __init__(self, train_config: TrainingConfig, melspec_config: MelSpectrogramConfig):
        self.config = train_config
        self.melspec_config = melspec_config

        self.melspec = MelSpectrogram(melspec_config, center=False).to(train_config.device)

        self.train_wavs = os.listdir(os.path.join(train_config.train_path, "wavs"))

    def __len__(self):
        return len(self.train_wavs)

    def melspec_with_pad(self, wav):
        pad_len = (self.melspec_config.n_fft - self.melspec_config.hop_length) // 2
        wav_data_for_mel = F.pad(wav, (pad_len, pad_len), mode='reflect')
        return self.melspec(wav_data_for_mel)
    
    def __getitem__(self, idx):
        wav_file = os.path.join(self.config.train_path, "wavs", self.train_wavs[idx])
        wav_data = torchaudio.load(wav_file)[0].to(self.config.device)

        if wav_data.size(1) < self.config.segment_length:
            wav_data = F.pad(wav_data, (0, self.config.segment_length - wav_data.size(1)))
        elif wav_data.size(1) > self.config.segment_length:
            start = random.randint(0, wav_data.size(1) - self.config.segment_length)
            wav_data = wav_data[:, start: start + self.config.segment_length]

        mel = self.melspec_with_pad(wav_data)
        return wav_data, mel
