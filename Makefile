.PHONY: submit

submit:
	@if [ -z "$(model_name)" ] || [ -z "$(model_type)" ]; then \
		echo "Error: model_name and model_type must be provided"; \
		echo "Usage: make submit model_name=... model_type=..."; \
		exit 1; \
	fi
	@echo "Evaluating model: $(model_name) with type: $(model_type)"
	python run.py configs/$(model_type)_fleurs.yaml --model_name=$(model_name)
	python run.py configs/$(model_type)_commonvoice.yaml --model_name=$(model_name)
	@model_cache_path=$$(echo $(model_name) | sed 's|/|--|g'); \
	model_full_path="$${HOME}/.cache/huggingface/hub/models--$$model_cache_path"; \
	if [ -d "$$model_full_path" ]; then \
		echo "Deleting model cache at: $$model_full_path"; \
		rm -rf "$$model_full_path"; \
	else \
		echo "Model cache not found: $$model_full_path"; \
	fi

