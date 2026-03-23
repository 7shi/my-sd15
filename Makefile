.PHONY: help download run samples clean

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
	@echo "  make download-all     - Download all models"
	@echo "  make download-sd15    - Download Stable Diffusion 1.5"
	@echo "  make download-minisd  - Download miniSD"
	@echo "  make download-any5    - Download Anything V5"
	@echo "  make download-lcm     - Download LCM LoRA"
	@echo "  make download-taesd   - Download Tiny AutoEncoder (TAESD)"
	@echo "  make run              - Run generation with default prompt"
	@echo "  make samples          - Generate sample images for all models (requires all weights)"
	@echo "  make clean            - Remove all downloaded weights"

download: download-sd15-tokenizer download-minisd

download-all: download-sd15 download-minisd download-any5 download-lcm download-taesd

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
	uv run single2dir.py --bits 16 weights/$(MINISD_ID)/$(MINISD_ST)

ANY5_ID = genai-archive/anything-v5
ANY5_ST = anything-v5.safetensors

download-any5:
	$(call download_model,$(ANY5_ID),$(ANY5_ST))
	uv run single2dir.py --bits 16 weights/$(ANY5_ID)/$(ANY5_ST)

LCM_ID = latent-consistency/lcm-lora-sdv1-5

download-lcm:
	$(call download_model,$(LCM_ID),pytorch_lora_weights.safetensors)

TAESD_ID = madebyollin/taesd

download-taesd:
	$(call download_model,$(TAESD_ID),config.json diffusion_pytorch_model.safetensors)

PROMPT  = -p "a cat sitting on a windowsill"
OPTIONS = $(PROMPT) --steps 10 --cfg 7.5

run:
	uv run my-sd15 $(OPTIONS)

SEED = --seed 123
SAMPLE_OPTS = $(OPTIONS) $(SEED)
LCM_OPTS = $(PROMPT) --steps 3 --cfg 1 $(SEED) --lcm --lora $(LCM_ID)
TAESD_OPTS = --vae $(TAESD_ID)

# 引数: (1)=model_id, (2)=出力プレフィックス
define generate_samples
	uv run my-sd15 -m $(1) $(SAMPLE_OPTS) -W 256 -H 256 -o samples/$(2)-256x256.jpg
	uv run my-sd15 -m $(1) $(SAMPLE_OPTS) -W 512 -H 512 -o samples/$(2)-512x512.jpg
	uv run my-sd15 -m $(1) $(SAMPLE_OPTS) $(TAESD_OPTS) -W 256 -H 256 -o samples/$(2)-taesd-256x256.jpg
	uv run my-sd15 -m $(1) $(SAMPLE_OPTS) $(TAESD_OPTS) -W 512 -H 512 -o samples/$(2)-taesd-512x512.jpg
	uv run my-sd15 -m $(1) $(LCM_OPTS) -W 256 -H 256 -o samples/$(2)-lcm-256x256.jpg
	uv run my-sd15 -m $(1) $(LCM_OPTS) -W 512 -H 512 -o samples/$(2)-lcm-512x512.jpg
	uv run my-sd15 -m $(1) $(LCM_OPTS) $(TAESD_OPTS) -W 256 -H 256 -o samples/$(2)-lcm-taesd-256x256.jpg
	uv run my-sd15 -m $(1) $(LCM_OPTS) $(TAESD_OPTS) -W 512 -H 512 -o samples/$(2)-lcm-taesd-512x512.jpg
endef

samples:
	$(call generate_samples,$(SD15_ID),sd15)
	$(call generate_samples,$(MINISD_ID),minisd)
	$(call generate_samples,$(ANY5_ID),any5)

clean:
	rm -rf weights
