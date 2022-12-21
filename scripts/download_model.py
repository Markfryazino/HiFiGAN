import wandb
import os

api = wandb.Api()

os.makedirs("data/download", exist_ok=True)
run = api.run("broccoliman/HiFiGAN/2uxswlif")
run.file("checkpoint_4000.pth.tar").download(root="data/download")
