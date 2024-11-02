import torch
# from ..medusa.model.modeling_llama_ssd_v1 import LlamaForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


# from transformers.models.llama.modeling_llama import LlamaForCausalLM 

import argparse

from kl_searching import LayerGroupSearching_KL
from thru_searching import LayerGroupSearching_thru
from bayesian_opt import LayerGroupSearching_bayesian_thru

import numpy as np

import re

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=False, help="Model name or path.", default='/root/MODELS/vicuna-7b-v1.3')
    parser.add_argument("--n_shot", type=int, required=False, help="Number of shots.", default=1)
    parser.add_argument("--task_name", type=str, required=False, help="Task name.", default='/root/DATASETS/cnn_dailymail')
    parser.add_argument("--num_sample", type=int, required=False, help="num sample", default=8)
    parser.add_argument("--num_group", type=int, required=False, help="draft group num", default=3)
    parser.add_argument("--min_group_layers", type=int, required=False, help="min layers of a draft group", default=6)
    parser.add_argument("--max_group_layers", type=int, required=False, help="max layers of a draft group", default=10)
    parser.add_argument("--searching_method", type=str, required=False, help="searching method", default="kl")
    parser.add_argument("--seed", type=int, required=False, help="seed", default=42)
    parser.add_argument("--max_new_tokens", type=int, required=False, help="max new tokens", default=512)
    parser.add_argument("--max_seq_length", type=int, required=False, help="max seq len", default=2048)
    parser.add_argument("--bayesian_itr", type=int, required=False, help="bayesian_itr", default=100)
    parser.add_argument("--ts_layer_level", action='store_true', required=False, help="same attn and mlp", default=False)
    parser.add_argument("--top_layers_level", action='store_true', required=False, help="whether to keep top layers or not", default=False)
    parser.add_argument("--top_layers_len", type=int, required=False, help="top layers to keep", default=12)
    

    args = parser.parse_args()

    torch.nn.Linear.reset_parameters = lambda x: None


    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用，使用 CPU")


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    seed =args.seed


    if args.searching_method == "kl":
        from modeling_llama_ssd_gp import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    elif args.searching_method == "thru" or args.searching_method == "bayesian":
        # from modeling_llama_ssd_v1 import LlamaForCausalLM
        # from modeling_llama_ssd_v3 import LlamaForCausalLM
        from modeling_llama_ssd_v1_top_layers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    task = re.search(r'[^/]+$', args.task_name).group()

    
    if args.searching_method == "kl":
        prompts = []

        if task == "cnn_dailymail":
            cnn = load_dataset(args.task_name, '3.0.0').shuffle(seed)
            for i in range(args.num_sample):
                item = cnn['train'][i+100]
                cnn_context = 'Article: ' + item['article'] + '\nSummary: ' + item['highlights'].replace('\n', '')
                
                item = cnn['train'][i]
                prompt = cnn_context + '\nArticle: ' + item['article'] + '\nSummary:'
                prompts.append(prompt)
                
        elif task == "xsum":
            xsum = load_dataset(args.task_name).shuffle(seed)
            for i in range(args.num_sample):
                item = xsum['train'][i+100]
                xsum_context = 'Article: ' + item['document'] + '\nSummary: ' + item['summary'].replace('\n', '')
                
                item = xsum['train'][i]
                prompt = xsum_context + '\nArticle: ' + item['document'] + '\nSummary:'
                prompts.append(prompt)

    elif args.searching_method == "thru" or args.searching_method == "bayesian":
        prompt_shots = ''
        if task == 'xsum':
            data = load_dataset(args.task_name, split='train').shuffle(seed=seed).select(range(args.num_sample))
            shots = load_dataset(args.task_name,split='train').shuffle(seed=seed).select(range(args.n_shot))
            prompt_keys=['document','summary']
        elif task == 'cnn_dailymail':
            data = load_dataset(args.task_name, name='3.0.0', split='train').shuffle(seed=seed).select(range(args.num_sample))
            shots = load_dataset(args.task_name, name='3.0.0', split='train').shuffle(seed=seed).select(range(args.n_shot))
            prompt_keys=['article','highlights']
        for i in range(args.n_shot):
            prompt = 'Article: ' + shots[i][prompt_keys[0]] + '\nSummary: ' + shots[i][prompt_keys[1]].replace('\n', '') + '\n'
            prompt_shots += prompt


    if args.searching_method == "kl":
        layer_group_searching = LayerGroupSearching_KL(model, tokenizer, prompts, args.num_group, args.min_group_layers, args.max_group_layers)
    elif args.searching_method == "thru":
        layer_group_searching = LayerGroupSearching_thru(model, tokenizer, data, prompt_shots, task, args.max_new_tokens, args.num_group, args.min_group_layers, args.max_group_layers)
    elif args.searching_method == "bayesian":
        layer_group_searching = LayerGroupSearching_bayesian_thru(model, tokenizer, data, prompt_shots, task, args.max_new_tokens, args.max_seq_length, args.bayesian_itr, args.num_group, args.min_group_layers, args.max_group_layers, args.ts_layer_level, args.top_layers_level, args.top_layers_len)
    

    # layer_group_searching.optimize_transformer_layer()
    layer_group_searching.optimize_attn_mlp()
    # layer_group_searching.optimize_attn_mlp_middle([0, 1, 2])
    # layer_group_searching.optimize_with_ortools()