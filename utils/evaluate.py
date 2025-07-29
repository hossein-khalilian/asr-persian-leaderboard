import os

from dotenv import load_dotenv
from jiwer import cer, wer
from text_normalizer.persian_normalizer import (
    normalize_persian_halfspace,
    persian_normalizer,
    persian_normalizer_no_punc,
)

load_dotenv()


def evaluate_asr(reference: list[str], hypothesis: list[str]) -> dict:
    """
    Compute WER and CER using the jiwer library after applying Persian normalization.
    """

    if os.environ.get("no_punctuation").lower() == "true":
        reference_normalized = [
            normalize_persian_halfspace(persian_normalizer_no_punc(ref))
            for ref in reference
        ]
        hypothesis_normalized = [
            normalize_persian_halfspace(persian_normalizer_no_punc(pred))
            for pred in hypothesis
        ]

        reference_normalized = [
            ref.replace("\u200c", " ") for ref in reference_normalized
        ]
        hypothesis_normalized = [
            pred.replace("\u200c", " ") for pred in hypothesis_normalized
        ]

    else:
        reference_normalized = [persian_normalizer(r) for r in reference]
        hypothesis_normalized = [persian_normalizer(h) for h in hypothesis]

    wer_score = wer(reference_normalized, hypothesis_normalized)
    cer_score = cer(reference_normalized, hypothesis_normalized)

    return {"wer": wer_score, "cer": cer_score}
