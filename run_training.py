from src.config import TrainingConfig, GeneratorConfig, DiscriminatorConfig, MelSpectrogramConfig
from src.train import prepare, train

args = prepare(TrainingConfig(eval_steps=10), MelSpectrogramConfig(), GeneratorConfig(), DiscriminatorConfig())
train(args)
