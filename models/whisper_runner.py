import logging
import time
import warnings
from datetime import date

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from utils.evaluate import evaluate_asr

warnings.simplefilter(action="ignore", category=FutureWarning)

# --------------------- Logging Setup ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_whisper(config):
    """
    Run Whisper ASR on a dataset split according to config.
    Returns dict with metrics and info.

    config keys:
      - model_name (str): Hugging Face model name (e.g. "openai/whisper-large-v3")
      - dataset (str): dataset name (e.g. "hsekhalilian/commonvoice")
      - split (str): dataset split (e.g. "dev")
      - language (str): language id for Whisper (e.g. "persian")
      - sample_size (int, optional): limit number of samples
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        hardware_info = f"{gpu_name} (CUDA {cuda_version})"
    else:
        hardware_info = "CPU only"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = config["model_name"]
    dataset_name = config["dataset"]
    split = config.get("split", "test")
    language = config.get("language", "english")
    sample_size = config.get("sample_size")

    parts = model_id.strip("/").split("/")
    model_name = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else parts[-1]

    logger.info(f"Loading model {model_id} on {device}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    # if hasattr(model.generation_config, "forced_decoder_ids"):
    #     del model.generation_config.forced_decoder_ids

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
    )

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(dataset_name, split=split)
    if sample_size:
        dataset = dataset.select(range(sample_size))
    sample_size = len(dataset)

    references = []
    predictions = []

    logger.info(f"Running inference on {sample_size} samples...")
    start_time = time.time()
    for sample in tqdm(dataset):
        result = pipe(
            sample["audio"],
            generate_kwargs={
                # "task": "transcribe",
                # "language": language,
            },
        )
        predictions.append(result["text"])
        ref = sample.get("normalized_transcription") or sample.get("sentence") or ""
        references.append(ref.lower().strip())

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
