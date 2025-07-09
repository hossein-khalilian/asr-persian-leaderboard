import json
import logging
import time
from datetime import date

import torch
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

from utils.create_dataset import create_nemo_dataset
from utils.evaluate import evaluate_asr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)


def run_nemo(config):
    """
    Run Wav2vec2 ASR on a dataset split according to config.
    Returns dict with metrics and info.

    config keys:
      - model_name (str): Hugging Face model name
      - dataset (str): dataset name (e.g. "common_voice")
      - subset (str): dataset subset (e.g. "fa")
      - split (str): dataset split (e.g. "test[:20]")
      - language (str): language id for Wav2vec2
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        hardware_info = f"{gpu_name} (CUDA {cuda_version})"
    else:
        hardware_info = "CPU only"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config["model_name"]

    logger.info(f"Loading model {model_name} on {device}...")
    asr_model = EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=model_name)
    asr_model = asr_model.to(device)
    asr_model.eval()

    # Load dataset
    manifest_path = create_nemo_dataset(config)
    references = []
    predictions = []
    audio_files = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            audio_files.append(entry["audio_filepath"])
            references.append(entry["text"])

    logger.info(f"Running inference on {len(audio_files)} samples...")
    start_time = time.time()

    results = asr_model.transcribe(audio_files)
    predictions = [result.text for result in results]

    elapsed_time = time.time() - start_time
    metrics = evaluate_asr(predictions, references)

    result = {
        "Rank": "",
        "Model Name": model_name,
        "WER (%)": round(metrics["wer"] * 100, 2),
        "CER (%)": round(metrics["cer"] * 100, 2),
        "Inference Time (s)": round(elapsed_time, 2),
        "Dataset Used": f"{config.get('dataset')} {config.get('split')}".strip(),
        "Sample Size": len(audio_files),
        "# Params (M)": round(asr_model.num_weights / 1e6, 2),
        "Hugging Face Link": f"https://huggingface.co/{model_name}",
        "Hardware Info": hardware_info,
        "Last Updated": str(date.today()),
        "Notes": config.get("notes", ""),
    }

    return result
