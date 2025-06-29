from datetime import date


def run_dummy(config):
    result = {
        "Rank": "",  # Will be recalculated
        "Model Name": "dummy_model",
        "WER (%)": 6.1,
        "CER (%)": 3.8,
        "Inference Time (s)": 0.75,
        "Dataset Used": "Common Voice (Persian)",
        "# Params (M)": 244,
        "Hugging Face Link": "https://huggingface.co/dummy/model",
        "Last Updated": str(date.today()),
        "Notes": "Zero-shot test run",
    }

    return result
