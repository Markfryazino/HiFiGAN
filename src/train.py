import torch
import torch.nn.functional as F
import numpy as np
import os
import wandb
import random

from dataclasses import dataclass
from collections import namedtuple
from tqdm import tqdm

from src.config import TrainingConfig, GeneratorConfig, DiscriminatorConfig, MelSpectrogramConfig
from src.models import Generator, TotalDiscriminator
from src.data import MelWavDataset
from src.mel import MelSpectrogram


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def collate(samples):
    return torch.cat([s[0] for s in samples], 0), torch.cat([s[1] for s in samples], 0)


def inference(generator, mel_path, log_path, max_wav_value, step, device, sample_rate):
    generator.eval()
    for file in os.listdir(mel_path):
        mel = torch.load(os.path.join(mel_path, file)).unsqueeze(0).to(device)
        wav = generator(mel).squeeze(0).detach().cpu()
        wav_alt = (wav * max_wav_value).to(torch.int16)
        wandb.log(
            {
                "test/" + file.replace(".pt", ""): wandb.Audio(wav, sample_rate),
                "test_alt/" + file.replace(".pt", ""): wandb.Audio(wav_alt, sample_rate),
            },
            step=step
        )
    generator.train()


@dataclass
class TrainingArgs:
    train_config: TrainingConfig = None
    melspec_config: MelSpectrogramConfig = None
    generator_config: GeneratorConfig = None
    discriminator_config: DiscriminatorConfig = None
    dataset: MelWavDataset = None
    generator: Generator = None
    discriminator: TotalDiscriminator = None
    generator_opt: torch.optim.AdamW = None
    discriminator_opt: torch.optim.AdamW = None
    generator_scheduler: torch.optim.lr_scheduler.ExponentialLR = None
    discriminator_scheduler: torch.optim.lr_scheduler.ExponentialLR = None
    dataloader: torch.utils.data.DataLoader = None


def prepare(train_config: TrainingConfig, melspec_config: MelSpectrogramConfig,
            generator_config: GeneratorConfig, discriminator_config: DiscriminatorConfig):
    set_random_seed(train_config.seed)
    os.makedirs(train_config.logs_path, exist_ok=True)

    dataset = MelWavDataset(train_config, melspec_config)

    generator = Generator(generator_config).to(train_config.device)
    discriminator = TotalDiscriminator(discriminator_config).to(train_config.device)
    generator.train()
    discriminator.train()

    generator_opt = torch.optim.AdamW(generator.parameters(), lr=train_config.lr, betas=train_config.adam_betas)
    discriminator_opt = torch.optim.AdamW(discriminator.parameters(), lr=train_config.lr, betas=train_config.adam_betas)

    generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(generator_opt, train_config.lr_decay)
    discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator_opt, train_config.lr_decay)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=train_config.batch_size,
        shuffle=True,
        # num_workers=4,
        # pin_memory=True,
        collate_fn=collate
    )

    return TrainingArgs(
        train_config=train_config,
        melspec_config=melspec_config,
        generator_config=generator_config,
        discriminator_config=discriminator_config,
        dataset=dataset,
        generator=generator,
        discriminator=discriminator,
        generator_opt=generator_opt,
        discriminator_opt=discriminator_opt,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        dataloader=dataloader
    )


def discriminator_loss(real_out, fake_out):
    real_loss = ((real_out - 1) ** 2).mean()
    fake_loss = (fake_out ** 2).mean()
    return real_loss + fake_loss


def generator_loss(real_mel, real_disc_out, real_disc_hidden, fake_mel, fake_disc_out, 
                   fake_disc_hidden, l1_gamma=45., gan_gamma=8., fm_gamma=8.):
    l1_loss = F.l1_loss(real_mel, fake_mel) * l1_gamma
    gan_loss = ((fake_disc_out - 1) ** 2).mean() * gan_gamma
    fm_loss = F.l1_loss(real_disc_hidden, fake_disc_hidden) * fm_gamma
    return l1_loss, gan_loss, fm_loss, l1_loss + gan_loss + fm_loss


def train(args: TrainingArgs):
    set_random_seed(args.train_config.seed)

    wandb.init(project=args.train_config.wandb_project)

    pbar = tqdm(total=args.train_config.epochs * len(args.dataloader))

    steps = 0

    total_disc_loss = total_gen_l1_loss = total_gen_gan_loss = total_gen_fm_loss = total_gen_full_loss = 0.

    for epoch in range(args.train_config.epochs):
        for real_wav, real_mel in args.dataloader:
            steps += 1

            fake_wav = args.generator(real_mel)
            fake_mel = args.dataset.melspec_with_pad(fake_wav)

            disc_real_output, _ = args.discriminator(real_wav)
            disc_fake_output, _ = args.discriminator(fake_wav)
            disc_loss = discriminator_loss(disc_real_output, disc_fake_output.detach())

            args.discriminator_opt.zero_grad()
            disc_loss.backward()
            args.discriminator_opt.step()

            disc_real_output, disc_real_hiddens = args.discriminator(real_wav)
            disc_fake_output, disc_fake_hiddens = args.discriminator(fake_wav)

            gen_l1_loss, gen_gan_loss, gen_fm_loss, gen_full_loss = generator_loss(
                real_mel, disc_real_output, disc_real_hiddens,
                fake_mel, disc_fake_output, disc_fake_hiddens, 
                args.train_config.l1_gamma, args.train_config.gan_gamma,
                args.train_config.fm_gamma
            )

            args.generator_opt.zero_grad()
            gen_full_loss.backward()
            args.generator_opt.step()

            total_disc_loss += disc_loss.item() / args.train_config.log_steps
            total_gen_l1_loss += gen_l1_loss.item() / args.train_config.log_steps
            total_gen_gan_loss += gen_gan_loss.item() / args.train_config.log_steps
            total_gen_fm_loss += gen_fm_loss.item() / args.train_config.log_steps
            total_gen_full_loss += gen_full_loss.item() / args.train_config.log_steps

            if steps % args.train_config.log_steps == 0:
                wandb.log(
                    {
                        "discriminator_loss": total_disc_loss,
                        "generator_l1_loss": total_gen_l1_loss,
                        "generator_gan_loss": total_gen_gan_loss,
                        "generator_fm_loss": total_gen_fm_loss,
                        "generator_full_loss": total_gen_full_loss,
                    },
                    step=steps
                )
                total_disc_loss = total_gen_l1_loss = total_gen_gan_loss = 0.
                total_gen_fm_loss = total_gen_full_loss = 0.

            if steps % args.train_config.save_steps == 0:
                save_path = os.path.join(args.train_config.logs_path, 'checkpoint_%d.pth.tar' % steps)
                torch.save({
                    'generator': args.generator.state_dict(), 
                    'discriminator': args.discriminator.state_dict()
                    }, save_path
                )
                wandb.save(save_path)
                print("save model at step %d ..." % steps)
            
            if steps % args.train_config.eval_steps == 0:
                inference(
                    args.generator, args.train_config.inference_path, args.train_config.logs_path,
                    args.train_config.max_wav_value, steps, args.train_config.device, args.train_config.sample_rate
                )
            
            pbar.update(1)

        args.discriminator_scheduler.step()
        args.generator_scheduler.step()
    
    pbar.finish()