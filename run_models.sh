#!/bin/bash

# Array of model_type:model_name pairs
models=(
  "nemo:hsekhalilian/FastConformer-Hybrid-Transducer-CTC-BPE"
  "nemo:hsekhalilian/Speech_To_Text_Finetuning_01"
  "nemo:hsekhalilian/Speech_To_Text_Finetuning_02"
  "nemo:hsekhalilian/Speech_To_Text_Finetuning_03"
  "nemo:hsekhalilian/Speech_To_Text_Finetuning_03_no_punc_with_encoder"
  "nemo:hsekhalilian/speech-to-text-rnnt-finetuned"
  "nemo:hsekhalilian/stt_fa_fastconformer_updated_tokenizer"
  "nemo:hsekhalilian/stt_fa_fastconformer_updated_tokenizer_01"
  "nemo:nvidia/stt_fa_fastconformer_hybrid_large"
  # Add more like:
  # "whisper:openai/whisper-large"
  # "torchaudio:torchaudio/wav2vec2-large"
)


models=(
    "wav2vec2:alifarokh/wav2vec2-xls-r-300m-fa"
    "wav2vec2:facebook/mms-1b-all"
    "wav2vec2:hsekhalilian/wav2vec2-custom-model"
    "wav2vec2:jonatasgrosman/wav2vec2-large-xlsr-53-persian"
    "wav2vec2:m3hrdadfi/wav2vec2-large-xlsr-persian-shemo"
    "wav2vec2:m3hrdadfi/wav2vec2-large-xlsr-persian-v2"
    "wav2vec2:m3hrdadfi/wav2vec2-large-xlsr-persian-v3"
    "nemo:nvidia/stt_fa_fastconformer_hybrid_large"
    "nemo:alifarokh/nemo-conformer-medium-fa"
    "seamless:facebook/hf-seamless-m4t-medium"
    "seamless:facebook/seamless-m4t-large"
    "seamless:facebook/seamless-m4t-medium"
    "seamless:facebook/seamless-m4t-v2-large"
    "whisper:MohammadGholizadeh/whisper-large-v3-persian-common-voice-17"
    "whisper:MohammadKhosravi/whisper-large-v3-Persian"
    "nemo:Neurai/NeuraSpeech_900h"
    "whisper:Neurai/NeuraSpeech_WhisperBase"
    "whisper:openai/whisper-base"
    "whisper:openai/whisper-large"
    "whisper:openai/whisper-large-v2"
    "whisper:openai/whisper-large-v3"
    "whisper:openai/whisper-large-v3-turbo"
    "whisper:openai/whisper-medium"
    "whisper:openai/whisper-small"
    "whisper:steja/whisper-large-persian"
    "whisper:distil-whisper/distil-large-v3"
    "whisper:distil-whisper/distil-large-v3.5"
)


for entry in "${models[@]}"; do
  IFS=":" read -r model_type model_name <<< "$entry"
  echo "Running: make submit model_type=$model_type model_name=$model_name"
  make submit model_type="$model_type" model_name="$model_name"
done
