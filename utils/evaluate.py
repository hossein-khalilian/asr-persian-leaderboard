from jiwer import wer, cer


def evaluate_asr(predictions, references):
    """
    Compute WER and CER using the jiwer library.
    """
    pred_text = "\n".join(predictions)
    ref_text = "\n".join(references)

    wer_score = wer(ref_text, pred_text)
    cer_score = cer(ref_text, pred_text)

    return {"wer": wer_score, "cer": cer_score}
