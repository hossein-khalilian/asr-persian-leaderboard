import logging
import time

import torch
import torchaudio
from datasets import load_dataset
from jiwer import wer
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# --------------------- Setup Logging ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --------------------- Configuration ---------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/user/.cache/models/m3hrdadfi/wav2vec2-large-xlsr-persian-v3/"
SAMPLE_SIZE = 1000
TARGET_SAMPLING_RATE = 16000
BATCH_SIZE = 8

logger.info(f"Using device: {DEVICE}")

# --------------------- Load Dataset ---------------------
logger.info("Loading dataset...")
dataset = load_dataset("hsekhalilian/commonvoice", split="dev")
dataset = dataset.select(range(SAMPLE_SIZE))
logger.info(f"Loaded {len(dataset)} samples.")

# --------------------- Load Model and Processor ---------------------
logger.info("Loading model and processor...")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
logger.info("Model and processor loaded successfully.")

# --------------------- Resampling Function ---------------------
resampler = torchaudio.transforms.Resample(
    orig_freq=48000, new_freq=TARGET_SAMPLING_RATE
)


def preprocess(batch):
    speech_arrays = []
    for audio in batch["audio"]:
        speech_array, sampling_rate = audio["array"], audio["sampling_rate"]
        if sampling_rate != TARGET_SAMPLING_RATE:
            speech_array = resampler(torch.tensor(speech_array)).numpy()
        speech_arrays.append(speech_array)

    inputs = processor(
        speech_arrays,
        sampling_rate=TARGET_SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = processor.batch_decode(predicted_ids)

    return {
        "predicted": [t.lower() for t in transcriptions],
        "reference": [s.lower() for s in batch["sentence"]],
    }


# --------------------- Run Batched Inference with map() ---------------------
logger.info("Starting batched inference with Hugging Face `map()`...")
start_time = time.time()

result_dataset = dataset.map(
    preprocess, batched=True, batch_size=BATCH_SIZE, remove_columns=dataset.column_names
)

end_time = time.time()
logger.info(f"Inference completed in {end_time - start_time:.2f} seconds.")

# --------------------- Compute WER ---------------------
error_rate = wer(result_dataset["reference"], result_dataset["predicted"])
logger.info(f"Word Error Rate (WER): {error_rate:.2%}")
