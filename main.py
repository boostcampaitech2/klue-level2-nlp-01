import hydra
from os import walk, path
from omegaconf import DictConfig
from src.train import train
from src.inference import inference_main as inference


def multimain(filename, dir_path):
    cfg_path = path.join(dir_path, "..", "config")
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    @hydra.main(config_path=cfg_path, config_name=filename)
    def sub_main(cfg: DictConfig):
        train(cfg)
        if not cfg.hyperparameters.set:
            inference(cfg)

    return sub_main()


@hydra.main(config_path=".", config_name="main_cfg")
def main(cfg: DictConfig):
    if cfg.multi_config:
        f = []
        dir_path = cfg.dir_path.code
        for (_, _, filenames) in walk(path.join(cfg.dir_path.code, "..", "config")):
            print(filenames)
            for filename in filenames:
                if filename[-5:] == ".yaml":
                    f.append(filename)
        for filename in f:
            multimain(filename[:-5], dir_path)
    else:
        train(cfg)
        if not cfg.hyperparameters.set:
            inference(cfg)


if __name__ == "__main__":
    main()
