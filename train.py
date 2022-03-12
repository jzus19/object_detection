from omegaconf import DictConfig, OmegaConf
import hydra
from src.train import train

@hydra.main(config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    train(cfg)

if __name__ == "__main__":
    main()