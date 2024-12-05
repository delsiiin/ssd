
CUDA_VISIBLE_DEVICES=1 WANDB_MODE="offline" accelerate launch -m axolotl.cli.train examples/ssd-vicuna-7b-v1.3/qlora_router_top_24_k_3_ee_stage_1_all.yml

CUDA_VISIBLE_DEVICES=1 WANDB_MODE="offline" accelerate launch -m axolotl.cli.train examples/ssd-vicuna-7b-v1.3/qlora_router_top_20_k_3_ee_stage_1_all.yml

# CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE="offline" accelerate launch -m axolotl.cli.train examples/ssd-llama-2/qlora_router_top_24_k_4_ee_stage_1_all.yml

# CUDA_VISIBLE_DEVICES=0 WANDB_MODE="offline" accelerate launch -m axolotl.cli.train examples/ssd/qlora_router_top_24_k_4.yml

# CUDA_VISIBLE_DEVICES=0 WANDB_MODE="offline" accelerate launch -m axolotl.cli.train examples/ssd/qlora_router_top_24_k_5.yml

# CUDA_VISIBLE_DEVICES=1 WANDB_MODE="offline" accelerate launch -m axolotl.cli.train examples/ssd-vicuna-7b-v1.3/qlora_router_top_16_k_4_ee_stage_1_all.yml

