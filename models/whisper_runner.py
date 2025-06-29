import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from utils.evaluate import evaluate_asr


def run_whisper(config):
    """
    Run Whisper ASR on a dataset split according to config.
    Returns dict with metrics and info.

    config keys:
      - model_name (str): Hugging Face model name
      - dataset (str): dataset name (e.g. "common_voice")
      - subset (str): dataset subset (e.g. "fa")
      - split (str): dataset split (e.g. "test[:20]")
      - language (str): language id for Whisper
    """
    model_name = config["model_name"]
    dataset_name = config["dataset"]
    subset = config.get("subset")
    split = config.get("split", "test")
    language = config.get("language", "en")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {model_name} on {device}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    # Load dataset
    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    references = []
    predictions = []

    print(f"Running inference on {len(ds)} samples...")
    for sample in ds:
        audio = sample["audio"]["array"] if "audio" in sample else sample["audio_array"]

        # Prepare inputs for whisper
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        # Set forced decoder token id to language id
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

        generated_ids = model.generate(inputs)
        transcription = (
            processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            .lower()
            .strip()
        )

        references.append(sample["text"].lower().strip())
        predictions.append(transcription)

    # Evaluate
    metrics = evaluate_asr(predictions, references)

    result = {
        "Model Name": model_name,
        "WER (%)": round(metrics["wer"] * 100, 2),
        "CER (%)": round(metrics["cer"] * 100, 2),
        "Dataset Used": f"{dataset_name} {subset or ''} {split}".strip(),
        "# Params (M)": round(model.num_parameters() / 1e6, 2),
        "Hugging Face Link": f"https://huggingface.co/{model_name}",
        "Last Updated": str(
            torch._C._get_tracing_state() is None
        ),  # just placeholder, you can put date.today()
        "Notes": config.get("notes", ""),
    }
    return result
