import logging
import time
import warnings

import torch
from datasets import load_dataset
from jiwer import wer
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.simplefilter(action="ignore", category=FutureWarning)

# --------------------- Setup Logging ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --------------------- Configuration ---------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
SAMPLE_SIZE = 100
logger.info(f"Using device: {DEVICE}")

# --------------------- Load Dataset ---------------------
logger.info("Loading dataset...")
dataset = load_dataset("hsekhalilian/commonvoice", split="dev")
dataset = dataset.select(range(SAMPLE_SIZE))
logger.info(f"Loaded {len(dataset)} samples.")

# --------------------- Load Model and Processor ---------------------
logger.info("Loading model and processor...")
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(DEVICE)
del model.generation_config.forced_decoder_ids

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=DEVICE,
)
logger.info("Model and processor loaded successfully.")

# --------------------- Inference ---------------------
logger.info("Starting inference...")
start_time = time.time()

predictions = []
references = []

for sample in tqdm(dataset):
    result = pipe(
        sample["audio"], generate_kwargs={"task": "transcribe", "language": "persian"}
    )
    predictions.append(result["text"])
    references.append(sample["normalized_transcription"])


end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"Inference completed in {elapsed_time:.2f} seconds.")

# --------------------- Compute WER ---------------------
error_rate = wer(references, predictions)
logger.info(f"Word Error Rate (WER): {error_rate:.2%}")
