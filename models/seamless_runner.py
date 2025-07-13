import logging
import time
from datetime import date

import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

from utils.evaluate import evaluate_asr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_seamless(config):
    """
    Run SeamlessM4T ASR on a dataset split according to config.
    Returns dict with metrics and info.

    config keys:
      - model_name (str): Hugging Face model name
      - dataset (str): dataset name (e.g. "common_voice")
      - subset (str): dataset subset (e.g. "fa")
      - split (str): dataset split (e.g. "test[:20]")
      - language (str): language id for SeamlessM4T
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        hardware_info = f"{gpu_name} (CUDA {cuda_version})"
    else:
        hardware_info = "CPU only"

    TARGET_SAMPLING_RATE = 16000
    model_name = config["model_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = config["dataset"]
    split = config.get("split", "test")
    sample_size = config.get("sample_size")

    logger.info(f"Loading model {model_name} on {device}...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        "facebook/seamless-m4t-v2-large"
    ).to(device)
    model.eval()

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    if sample_size:
        dataset = dataset.select(range(sample_size))
    sample_size = len(dataset)

    references = []
    predictions = []

    logger.info(f"Running inference on {sample_size} samples...")
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

        audio_inputs = processor(
            audios=speech_array, return_tensors="pt", sampling_rate=TARGET_SAMPLING_RATE
        ).to(device)

        output_tokens = model.generate(**audio_inputs, tgt_lang="pes")
        transcription = processor.decode(
            output_tokens[0].tolist(), skip_special_tokens=True
        )

        ref = sample.get("normalized_transcription") or sample.get("sentence") or ""
        references.append(ref.lower().strip())
        predictions.append(transcription)

    elapsed_time = time.time() - start_time
    metrics = evaluate_asr(predictions, references)

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


def run_seamless_batched(config):
    """
    Run SeamlessM4T ASR on a dataset split according to config.
    Returns dict with metrics and info.

    config keys:
      - model_name (str): Hugging Face model name
      - dataset (str): dataset name (e.g. "common_voice")
      - subset (str): dataset subset (e.g. "fa")
      - split (str): dataset split (e.g. "test[:20]")
      - language (str): language id for SeamlessM4T
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        hardware_info = f"{gpu_name} (CUDA {cuda_version})"
    else:
        hardware_info = "CPU only"

    TARGET_SAMPLING_RATE = 16000
    model_name = config["model_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = config["dataset"]
    split = config.get("split", "test")
    sample_size = config.get("sample_size")
    batch_size = config.get("batch_size")

    logger.info(f"Loading model {model_name} on {device}...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        "facebook/seamless-m4t-v2-large"
    ).to(device)
    model.eval()

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    if sample_size:
        dataset = dataset.select(range(sample_size))
    sample_size = len(dataset)

    def collate_fn(batch):
        speech_arrays = [sample["audio"]["array"] for sample in batch]
        sentences = []
        for sample in batch:
            sentence = (
                sample.get("normalized_transcription") or sample.get("sentence") or ""
            )
            sentences.append(sentence)

        return speech_arrays, sentences

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    references = []
    predictions = []

    logger.info(
        f"Running inference on {sample_size} samples with batch_size {batch_size}..."
    )
    start_time = time.time()

    for speech_arrays, sentences in tqdm(loader):
        audio_inputs = processor(
            audios=speech_arrays,
            return_tensors="pt",
            sampling_rate=TARGET_SAMPLING_RATE,
            padding=True,
        ).to(device)

        output_tokens = model.generate(**audio_inputs, tgt_lang="pes")
        transcriptions = processor.batch_decode(output_tokens, skip_special_tokens=True)

        references.extend(sentences)
        predictions.extend(transcriptions)

    elapsed_time = time.time() - start_time
    metrics = evaluate_asr(predictions, references)

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
