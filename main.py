import hydra
from omegaconf import DictConfig
from src.train import train
from src.inference import inference_main as inference


@hydra.main(config_path="conf", config_name="main")
def main(cfg: DictConfig):
    train(cfg)
    if not cfg.hyperparameters.set:
        inference(cfg)


if __name__ == "__main__":
    main()
