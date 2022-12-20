from src.config import TrainingConfig, GeneratorConfig, DiscriminatorConfig, MelSpectrogramConfig
from src.train import prepare, train

args = prepare(TrainingConfig(), MelSpectrogramConfig(), GeneratorConfig(), DiscriminatorConfig())
train(args)
