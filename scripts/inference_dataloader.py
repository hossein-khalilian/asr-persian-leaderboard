import logging
import time

import torch
import torchaudio
from datasets import load_dataset
from jiwer import wer
from torch.utils.data import DataLoader, Dataset
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
BATCH_SIZE = 12

logger.info(f"Using device: {DEVICE}")

# --------------------- Load Dataset ---------------------
logger.info("Loading dataset...")
dataset = load_dataset("hsekhalilian/commonvoice", split="dev")
dataset = dataset.select(range(SAMPLE_SIZE))
logger.info(f"Loaded {len(dataset)} samples.")

# --------------------- Preprocess Audio ---------------------
logger.info("Preprocessing audio data...")

resampler = torchaudio.transforms.Resample(
    orig_freq=48000, new_freq=TARGET_SAMPLING_RATE
)

speech_arrays = []
references = []

for sample in tqdm(dataset, desc="Resampling"):
    audio = sample["audio"]
    speech_array, sampling_rate = audio["array"], audio["sampling_rate"]

    # Resample if needed
    if sampling_rate != TARGET_SAMPLING_RATE:
        resampled = resampler(torch.tensor(speech_array)).numpy()
    else:
        resampled = speech_array

    speech_arrays.append(resampled)
    references.append(sample["sentence"].lower())


# --------------------- Dataset and DataLoader ---------------------
class SpeechDataset(Dataset):
    def __init__(self, speech_arrays, references):
        self.speech_arrays = speech_arrays
        self.references = references

    def __len__(self):
        return len(self.speech_arrays)

    def __getitem__(self, idx):
        return self.speech_arrays[idx], self.references[idx]


def collate_fn(batch):
    # Separate audio and reference
    audios, refs = zip(*batch)
    return list(audios), list(refs)


speech_dataset = SpeechDataset(speech_arrays, references)
data_loader = DataLoader(
    speech_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
)

# --------------------- Load Model and Processor ---------------------
logger.info("Loading model and processor...")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
logger.info("Model and processor loaded successfully.")

# --------------------- Inference ---------------------
logger.info("Starting inference...")
start_time = time.time()

all_predictions = []
all_references = []

for batch_audio, batch_ref in tqdm(data_loader, desc="Running inference"):
    inputs = processor(
        batch_audio,
        sampling_rate=TARGET_SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predictions = processor.batch_decode(predicted_ids)

    all_predictions.extend([pred.lower() for pred in predictions])
    all_references.extend(batch_ref)

end_time = time.time()
logger.info(f"Inference completed in {end_time - start_time:.2f} seconds.")

# --------------------- Compute WER ---------------------
error_rate = wer(all_references, all_predictions)
logger.info(f"Word Error Rate (WER): {error_rate:.2%}")
