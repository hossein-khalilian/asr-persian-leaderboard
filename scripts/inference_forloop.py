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

# --------------------- Inference ---------------------
logger.info("Starting inference...")
start_time = time.time()

predictions = []
references = []

for i, sample in enumerate(tqdm(dataset, desc="Processing")):
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
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # Store results
    predictions.append(transcription.lower())
    references.append(sample["sentence"].lower())

end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"Inference completed in {elapsed_time:.2f} seconds.")

# --------------------- Compute WER ---------------------
error_rate = wer(references, predictions)
logger.info(f"Word Error Rate (WER): {error_rate:.2%}")
