
CUDA_VISIBLE_DEVICES=0 WANDB_MODE="offline" accelerate launch -m axolotl.cli.train examples/ssd/qlora_router_top_24_k_3.yml

CUDA_VISIBLE_DEVICES=0 WANDB_MODE="offline" accelerate launch -m axolotl.cli.train examples/ssd/qlora_router_top_24_k_2.yml

CUDA_VISIBLE_DEVICES=0 WANDB_MODE="offline" accelerate launch -m axolotl.cli.train examples/ssd/qlora_router_top_24_k_4.yml

CUDA_VISIBLE_DEVICES=0 WANDB_MODE="offline" accelerate launch -m axolotl.cli.train examples/ssd/qlora_router_top_24_k_5.yml

CUDA_VISIBLE_DEVICES=0 WANDB_MODE="offline" accelerate launch -m axolotl.cli.train examples/ssd/qlora_router_top_24_k_6.yml