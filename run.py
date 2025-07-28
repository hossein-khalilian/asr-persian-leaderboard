import argparse
import importlib
import os

import yaml
from dotenv import load_dotenv

from utils.leaderboard import update_leaderboard

load_dotenv()


def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def import_runner(runner_name):
    """
    Dynamically import the runner function based on the name.
    Expects 'runner_name' to be in the format 'module_name.function_name',
    e.g., 'models.wav2vec2_runner.run_wav2vec2'.
    """
    try:
        module_path, func_name = runner_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import runner '{runner_name}': {e}")


def main(config_path):
    config = load_config(config_path)

    runner_name = config.get("model_runner")
    if not runner_name:
        raise ValueError("Config must specify 'model_runner' as full import path")

    benchmark_function = import_runner(runner_name)
    result = benchmark_function(config)

    if bool(os.environ.get("no_punctuation")):
        update_leaderboard(result, csv_path="leaderboard_nopunc.csv")
    else:
        update_leaderboard(result, csv_path="leaderboard.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark with specified model runner."
    )
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)
