import comet_ml  # noqa
import hydra
from omegaconf import DictConfig, OmegaConf
from vital.runner import VitalRunner


class Runner(VitalRunner):
    """Abstract runner that runs the main training/val loop, etc. using Lightning Trainer."""

    @classmethod
    def pre_run_routine(cls) -> None:
        """Sets-up the environment before running the training/testing."""
        super().pre_run_routine()

        OmegaConf.register_new_resolver(
            "camus_labels", lambda x: '-' + '-'.join([n for n in x if n != 'BG']) if len(x) != 4 else ''
        )
        OmegaConf.register_new_resolver(
            "frac", lambda x: int(x*100)
        )

    @staticmethod
    @hydra.main(config_path="config", config_name="default.yaml")
    def run_system(cfg: DictConfig) -> None:
        """Handles the training and evaluation of a model.

        Redefined to add @hydra.main decorator with correct config_path and config_name
        """
        return VitalRunner.run_system(cfg)


if __name__ == "__main__":
    Runner.main()
