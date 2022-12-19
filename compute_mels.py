import argparse
import os
import torch
import torchaudio

from tqdm import tqdm

from src.mel import MelSpectrogram
from src.config import MelSpectrogramConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset containing 'wavs' folder")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.dataset, "mels"), exist_ok=True)

    mel_spec = MelSpectrogram(MelSpectrogramConfig())

    wavs = os.listdir(os.path.join(args.dataset, "wavs"))
    for wav in tqdm(wavs):
        wav_data = torchaudio.load(os.path.join(args.dataset, "wavs", wav))[0]
        mel = mel_spec(wav_data).squeeze(0)
        torch.save(mel, os.path.join(args.dataset, "mels", wav.replace(".wav", ".pt")))


if __name__ == "__main__":
    main()
