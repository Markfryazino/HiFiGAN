import argparse
import os
import torch
import json
import wandb

from scipy.io.wavfile import write

from src.models import Generator
from src.config import GeneratorConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/repos/tts_project/data/download/checkpoint_4000.pth.tar", help="Path to model checkpoint")
    parser.add_argument("--mels", type=str, default="/repos/tts_project/data/MarkovkaSpeech/mels", help="Path to directory with melspectrograms")
    parser.add_argument("--save-path", type=str, default="/repos/tts_project/data/MarkovkaSpeech/generated", help="Path to directory with melspectrograms")
    parser.add_argument("--device", default="cuda:0", help="Device for synthesis")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    model = Generator(GeneratorConfig()).to(args.device)
    model.load_state_dict(torch.load(args.model)["generator"])

    model.eval()
    for file in os.listdir(args.mels):
        mel = torch.load(os.path.join(args.mels, file)).unsqueeze(0).to(args.device)
        wav = model(mel).squeeze(0).detach().cpu().numpy()
        write(os.path.join(args.save_path, file.replace(".pt", ".wav")), 22050, wav)


if __name__ == "__main__":
    main()
