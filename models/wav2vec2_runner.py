import logging
import time
import warnings
from datetime import date

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor

from utils.evaluate import evaluate_asr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_wav2vec2(config):
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

    TARGET_SAMPLING_RATE = 16000

    model_name_or_path = config["model_name"]
    parts = model_name_or_path.strip("/").split("/")
    model_name = f"{parts[-2]}/{parts[-1]}"

    dataset_name = config["dataset"]
    split = config.get("split", "test")
    sample_size = config.get("sample_size")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model {model_name} on {device}...")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Passing `gradient_checkpointing` to a config initialization.*",
        )
        model_config = Wav2Vec2Config.from_pretrained(model_name_or_path)
        processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        if hasattr(model_config, "gradient_checkpointing"):
            del model_config.gradient_checkpointing

    model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path, config=model_config)
    model = model.to(device)
    model.gradient_checkpointing_disable()
    model.eval()

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    if sample_size:
        dataset = dataset.select(range(sample_size))
    sample_size = len(dataset)

    references = []
    predictions = []

    logger.info(f"Running inference on {len(dataset)} samples...")
    start_time = time.time()
    for sample in tqdm(dataset):
        audio = sample["audio"]
        speech_array, sampling_rate = audio["array"], audio["sampling_rate"]

        # Resample if needed
        if sampling_rate != TARGET_SAMPLING_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate, new_freq=TARGET_SAMPLING_RATE
            )
            speech_array = resampler(torch.tensor(speech_array)).numpy()

        # Preprocess
        inputs = processor(
            speech_array,
            sampling_rate=TARGET_SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Inference
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        ref = sample.get("normalized_transcription") or sample.get("sentence") or ""
        references.append(ref.lower().strip())
        predictions.append(transcription)

    elapsed_time = time.time() - start_time
    metrics = evaluate_asr(references, predictions)

    result = {
        "Rank": "",
        "Model Name": model_name,
        "WER (%)": round(metrics["wer"] * 100, 2),
        "CER (%)": round(metrics["cer"] * 100, 2),
        "Inference Time (s)": round(elapsed_time, 2),
        "Dataset Used": f"{dataset_name} {split}".strip(),
        "Sample Size": sample_size,
        "# Params (M)": round(model.num_parameters() / 1e6, 2),
        "Hugging Face Link": f"https://huggingface.co/{model_name}",
        "Hardware Info": hardware_info,
        "Last Updated": str(date.today()),
        "Notes": config.get("notes", ""),
    }

    return result
