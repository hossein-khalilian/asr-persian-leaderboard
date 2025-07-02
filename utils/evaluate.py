from jiwer import cer, wer

from utils.normalizer import persian_normalizer


def evaluate_asr(reference: list[str], hypothesis: list[str]) -> dict:
    """
    Compute WER and CER using the jiwer library after applying Persian normalization.
    """

    reference_normalized = [persian_normalizer(r) for r in reference]
    hypothesis_normalized = [persian_normalizer(h) for h in hypothesis]

    wer_score = wer(reference_normalized, hypothesis_normalized)
    cer_score = cer(reference_normalized, hypothesis_normalized)

    return {"wer": wer_score, "cer": cer_score}
