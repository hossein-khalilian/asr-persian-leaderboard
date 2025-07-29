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
	@echo "Updating leaderboard..."
	python scripts/update_leaderboard.py --model_name=$(model_name) --model_type=$(model_type)

