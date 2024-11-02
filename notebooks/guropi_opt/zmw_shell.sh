
# attn mlp level (itr)
python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian  \
        --bayesian_itr 100 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --top_layers_len 20

python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian  \
        --bayesian_itr 200 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --top_layers_len 20

python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian  \
        --bayesian_itr 300 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --top_layers_len 20

python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian  \
        --bayesian_itr 400 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --top_layers_len 20

# python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian --ts_layer_level \
#         --bayesian_itr 500 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --top_layers_len 24

# python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian --ts_layer_level \
#         --bayesian_itr 600 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --top_layers_len 24

# attn mlp level (top len)
# CUDA_VISIBLE_DEVICES=1 python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian \
#         --bayesian_itr 200 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --top_layers_len 12

# CUDA_VISIBLE_DEVICES=1 python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian \
#         --bayesian_itr 200 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --top_layers_len 8

# CUDA_VISIBLE_DEVICES=1 python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian \
#         --bayesian_itr 200 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --top_layers_len 16

# CUDA_VISIBLE_DEVICES=1 python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian \
#         --bayesian_itr 200 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --top_layers_len 24

# CUDA_VISIBLE_DEVICES=1 python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian \
#         --bayesian_itr 200 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --top_layers_len 4


# ts layer level
# CUDA_VISIBLE_DEVICES=1 python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian \
#         --bayesian_itr 200 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --ts_layer_level --top_layers_len 12

# CUDA_VISIBLE_DEVICES=1 python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian \
#         --bayesian_itr 200 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --ts_layer_level --top_layers_len 12

# CUDA_VISIBLE_DEVICES=1 python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian \
#         --bayesian_itr 200 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --ts_layer_level --top_layers_len 12

# CUDA_VISIBLE_DEVICES=1 python run_searching.py --num_group 4 --num_sample 8 --searching_method bayesian \
#         --bayesian_itr 200 --max_seq_length 2048 --max_new_tokens 32 --top_layers_level --ts_layer_level --top_layers_len 12