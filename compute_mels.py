import argparse
import os
import torch
import torchaudio

from src.mel import MelSpectrogram
from src.config import MelSpectrogramConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset containing 'wavs' folder")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.dataset, "mels"), exist_ok=True)