.PHONY: download run clean help

# 引数: (1)=model_id, (2)=files
define download_model
	@mkdir -p weights/$(1)
	@BASE=https://huggingface.co/$(1)/resolve/main; \
	dl() { f=weights/$(1)/$$1; mkdir -p $$(dirname $$f); if [ -f "$$f" ]; then echo "  skip: $$f"; else curl -L -o "$$f" "$$BASE/$$1"; fi; }; \
	for f in $(2); do dl $$f; done
endef

help:
	@echo "Usage:"
	@echo "  make download         - Download minimal set (SD 1.5 tokenizer and miniSD)"
	@echo "  make download-sd15    - Download Stable Diffusion 1.5"
	@echo "  make download-minisd  - Download miniSD"
	@echo "  make download-any5    - Download Anything V5"
	@echo "  make run              - Run generation with default prompt"
	@echo "  make clean            - Remove all downloaded weights"

download: download-sd15-tokenizer download-minisd

SD15_ID = stable-diffusion-v1-5/stable-diffusion-v1-5

download-sd15: download-sd15-tokenizer download-sd15-weights

download-sd15-tokenizer:
	$(call download_model,$(SD15_ID),tokenizer/vocab.json tokenizer/merges.txt)

download-sd15-weights:
	$(call download_model,$(SD15_ID),text_encoder/model.fp16.safetensors unet/diffusion_pytorch_model.fp16.safetensors vae/diffusion_pytorch_model.fp16.safetensors)

MINISD_ID = webui/miniSD
MINISD_ST = miniSD.safetensors

download-minisd:
	$(call download_model,$(MINISD_ID),$(MINISD_ST))
	uv run single2dir.py --fp16 weights/$(MINISD_ID)/$(MINISD_ST)

ANY5_ID = genai-archive/anything-v5
ANY5_ST = anything-v5.safetensors

download-any5:
	$(call download_model,$(ANY5_ID),$(ANY5_ST))
	uv run single2dir.py --fp16 weights/$(ANY5_ID)/$(ANY5_ST)

OPTIONS = -p "a cat sitting on a windowsill" --seed 123 --steps 10 --cfg 7.5

run:
	#uv run my-sd15 $(OPTIONS) -W 256 -H 256 -o samples/sd15-256x256.jpg
	#uv run my-sd15 $(OPTIONS) -W 512 -H 512 -o samples/sd15-512x512.jpg
	uv run my-sd15 -m webui/miniSD $(OPTIONS) -W 256 -H 256 -o samples/minisd-256x256.jpg
	uv run my-sd15 -m webui/miniSD $(OPTIONS) -W 512 -H 512 -o samples/minisd-512x512.jpg
	#uv run my-sd15 -m genai-archive/anything-v5 $(OPTIONS) -W 256 -H 256 -o samples/any5-256x256.jpg
	#uv run my-sd15 -m genai-archive/anything-v5 $(OPTIONS) -W 512 -H 512 -o samples/any5-512x512.jpg

clean:
	rm -rf weights
