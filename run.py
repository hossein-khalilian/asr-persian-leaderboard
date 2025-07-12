import argparse

import yaml

from models.dolphin_runner import run_dolphin
from models.dummy_runner import run_dummy
from models.nemo_runner import run_nemo
from models.wav2vec2_runner import run_wav2vec2
from models.whisper_runner import run_whisper
from utils.leaderboard import update_leaderboard

RUNNERS = {
    "run_dummy": run_dummy,
    "run_wav2vec2": run_wav2vec2,
    "run_whisper": run_whisper,
    "run_nemo": run_nemo,
    "run_dolphin": run_dolphin,
}


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    runner_name = config.get("model_runner")
    if not runner_name:
        raise ValueError("Config must specify 'model_runner'")

    benchmark_function = RUNNERS[runner_name]
    result = benchmark_function(config)
    update_leaderboard(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
