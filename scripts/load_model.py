from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Config

model_name = "m3hrdadfi/wav2vec2-large-xlsr-persian-v3"
config = Wav2Vec2Config.from_pretrained(model_name)
del config.gradient_checkpointing

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name, config=config)

model_path = f"/home/user/.cache/models/{model_name}"
model.save_pretrained(model_path)
processor.save_pretrained(model_path)
print(f"model saved to {model_path} succcessfully.")