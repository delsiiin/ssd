python heads_accuracy.py --model_path /home/zmw/vicuna-7b-v1.3 --model_name vicuna-7b-v1.3 --data_path ./alpaca_eval.json \
                        --top_layers_len 24 --top_k_group 4 --early_exit

python gen_results.py --accuracy-path ./data/vicuna-7b-v1.3-24-4_heads_accuracy.pt --output-path ./data/graph.jpg