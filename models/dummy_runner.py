from datetime import date

import torch


def run_dummy(config):
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        hardware_info = f"{gpu_name} (CUDA {cuda_version})"
    else:
        hardware_info = "CPU only"

    result = {
        "Rank": "",  # Will be recalculated
        "Model Name": "dummy_model",
        "WER (%)": 4.8,
        "CER (%)": 3.8,
        "Inference Time (s)": 93,
        "Dataset Used": "Common Voice (Persian)",
        "Sample Size": 20,
        "# Params (M)": 244,
        "Hardware Info": hardware_info,
        "Hugging Face Link": "https://huggingface.co/dummy/model",
        "Last Updated": str(date.today()),
        "Notes": "Zero-shot test run",
    }

    return result
