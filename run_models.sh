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

for entry in "${models[@]}"; do
  IFS=":" read -r model_type model_name <<< "$entry"
  echo "Running: make submit model_type=$model_type model_name=$model_name"
  make submit model_type="$model_type" model_name="$model_name"
done
