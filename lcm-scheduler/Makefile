OPTIONS := -m stable-diffusion-v1-5/stable-diffusion-v1-5 \
           -p "a cat sitting on a windowsill" \
           --seed 123 --lora latent-consistency/lcm-lora-sdv1-5 \
           --lcm --steps 4 --cfg 1 -W 512 -H 512

SCHEDULER := ../my_sd15/scheduler.py

all: lcm-1.jpg lcm-2.jpg lcm-3.jpg

lcm-1.jpg: scheduler-1.py
	cp scheduler-1.py $(SCHEDULER)
	uv run my-sd15 $(OPTIONS) -o $@
	git checkout $(SCHEDULER)

lcm-2.jpg: scheduler-2.py
	cp scheduler-2.py $(SCHEDULER)
	uv run my-sd15 $(OPTIONS) -o $@
	git checkout $(SCHEDULER)

lcm-3.jpg:
	uv run my-sd15 $(OPTIONS) -o $@

.PHONY: all
