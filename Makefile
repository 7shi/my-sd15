.PHONY: download download-gpt2 download-rinna run clean help

SD15_ID = stable-diffusion-v1-5/stable-diffusion-v1-5

# 引数: (1)=model_id, (2)=files
define download_model
	@mkdir -p weights/$(1)
	@BASE=https://huggingface.co/$(1)/resolve/main; \
	dl() { f=weights/$(1)/$$1; mkdir -p $$(dirname $$f); if [ -f "$$f" ]; then echo "  skip: $$f"; else curl -L -o "$$f" "$$BASE/$$1"; fi; }; \
	for f in $(2); do dl $$f; done
endef

help:
	@echo "Usage:"
	@echo "  make download        - Download both models"
	@echo "  make download-sd15   - Download Stable Diffusion 1.5"
	@echo "  make run             - Run generation with default prompt"
	@echo "  make clean           - Remove all downloaded weights"

download: download-sd15

download-sd15:
	$(call download_model,$(SD15_ID),v1-5-pruned-emaonly.safetensors tokenizer/vocab.json tokenizer/merges.txt)
	uv run single2dir.py weights/$(SD15_ID)/v1-5-pruned-emaonly.safetensors

run:
	uv run my-sd15 --prompt "a cat sitting on a windowsill" --seed 42 --steps 10 --cfg 7.5 -o output.png

clean:
	rm -rf weights
