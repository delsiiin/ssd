
python test_vanilla.py --num_sample 10 --max_new_tokens 128 --model /home/zmw/vicuna-7b-v1.3

python test_medusa_serial.py --num_sample 10 --max_new_tokens 128 --model /home/zmw/medusa-vicuna-7b-v1.3

python test_router_serial.py --num_sample 10 --max_new_tokens 128 --top_layers_len 24 --top_k_group 4 --model /home/zmw/vicuna-7b-v1.3

python test_router_serial.py --num_sample 10 --max_new_tokens 128 --top_layers_len 24 --top_k_group 4 --model /home/zmw/vicuna-7b-v1.3 --davm
