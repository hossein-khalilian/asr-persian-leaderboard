import argparse
import importlib
import os

from dotenv import load_dotenv
from omegaconf import OmegaConf

from utils.leaderboard import update_leaderboard

load_dotenv()


def import_runner(runner_name):
    module_path, func_name = runner_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def main(config_path, model_name_override=None):
    cfg = OmegaConf.load(config_path)

    if model_name_override:
        cfg.model_name = model_name_override

    if OmegaConf.is_missing(cfg, "model_name"):
        raise ValueError("model_name is required but missing or marked ??? in config")

    benchmark_function = import_runner(cfg.model_runner)
    result = benchmark_function(cfg)

    csv_name = (
        "leaderboards/leaderboard_nopunc.csv"
        if os.environ.get("no_punctuation").lower() == "true"
        else "leaderboards/leaderboard.csv"
    )
    update_leaderboard(result, csv_path=csv_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark with YAML config")
    parser.add_argument(
        "config", help="Path to YAML config file (e.g. configs/myconfig.yaml)"
    )
    parser.add_argument(
        "--model_name", help="Override model_name in config", default=None
    )
    args = parser.parse_args()

    main(args.config, model_name_override=args.model_name)
