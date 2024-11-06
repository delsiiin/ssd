import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # define GPU id, remove if you want to use all GPUs available
import torch
from contextlib import contextmanager
import numpy as np
# from medusa.model.modeling_llama_ssd_v1 import LlamaForCausalLM
# from medusa.model.modeling_llama_ssd_v1_top_layers import LlamaForCausalLM
# from medusa.model.modeling_llama_ssd_router import LlamaForCausalLM, add_router
from transformers import LlamaForCausalLM
# from medusa.model.modeling_llama_ssd_v3 import LlamaForCausalLM
# from transformers.models.llama.modeling_llama import LlamaForCausalLM 
from medusa.model.configuration_llama_ssd import LlamaConfig
from medusa.model.kv_cache import *
from medusa.model.utils import *

from copy import deepcopy
import matplotlib.pyplot as plt

from transformers import top_k_top_p_filtering
from transformers import AutoTokenizer

from datasets import load_dataset

import json

import argparse

from tqdm import tqdm

from peft import PeftModel, PeftConfig

import time

from utils import *

from rouge_score import rouge_scorer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=False, help="Model name or path.", default='/root/MODELS/vicuna-7b-v1.3')
    parser.add_argument("--n_shot", type=int, required=False, help="Number of shots.", default=1)
    parser.add_argument("--task_name", type=str, required=False, help="Task name.", default='cnndm')
    parser.add_argument("--max_new_tokens", type=int, required=False, help="Max new tokens.", default=512)
    parser.add_argument("--max_seq_length", type=int, required=False, help="max seq len", default=2048)
    parser.add_argument("--do_sample", type=bool, required=False, help="Do sample.", default=False)
    parser.add_argument("--top_k", type=int, required=False, help="Top k.", default=50)
    parser.add_argument("--top_p", type=float, required=False, help="Top p.", default=0.8)
    parser.add_argument("--temperature", type=float, required=False, help="Temperature.", default=0.7)
    parser.add_argument("--seed", type=int, required=False, help="Seed.", default=42)
    parser.add_argument("--posterior_threshold", type=float, required=False, help="Posterior threshold.", default=0.09)
    parser.add_argument("--posterior_alpha", type=float, required=False, help="Posterior alpha.", default=0.3)
    parser.add_argument("--num_sample", type=int, required=False, help="Num sample.", default=10)
    parser.add_argument("--top_layers_len", type=int, required=False, help="top layers to keep", default=20)
    parser.add_argument("--top_k_group", type=int, required=False, help="Draft group num.", default=4)
    parser.add_argument("--resnet_num", type=int, required=False, help="resnet num.", default=1)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    seed =args.seed

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用，使用 CPU")

    n_shot = args.n_shot
    task_name = args.task_name
    prompt_shots = ''
    if task_name == 'xsum':
        data = load_dataset('/root/DATASETS/xsum', split='test').shuffle(seed=seed).select(range(1000))
        shots = load_dataset('/root/DATASETS/xsum',split='train').shuffle(seed=seed).select(range(n_shot))
        prompt_keys=['document','summary']
    elif task_name == 'cnndm':
        data = load_dataset('/root/DATASETS/cnn_dailymail', name='3.0.0', split='test') .shuffle(seed=seed).select(range(1000))
        shots = load_dataset('/root/DATASETS/cnn_dailymail', name='3.0.0', split='train').shuffle(seed=seed).select(range(n_shot))
        prompt_keys=['article','highlights']
    for i in range(n_shot):
        prompt = 'Article: ' + shots[i][prompt_keys[0]] + '\nSummary: ' + shots[i][prompt_keys[1]].replace('\n', '') + '\n'
        prompt_shots += prompt


    model_name = args.model

    # print(config.num_skipped_draft_model)

    # lora_path = "/root/idea/speculative_decoding/Medusa/axolotl/vicuna-7b-v1.3-qlora-ssd-out-router-top-24-k-5-seq-1024"
    # lora_config = PeftConfig.from_pretrained(lora_path)

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model = model.eval().to(device)

    # add_router(model)

    # model = PeftModel.from_pretrained(model, lora_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # total_avg_accept_length = 0

    start_time = time.time()

    with torch.no_grad():

        all_rouge_score = []

        for idx,x in tqdm(enumerate(data)):

            if idx == args.num_sample:
                end_time = time.time()
                break

            input_ids = clip_input(tokenizer, x, task_name, max_new_tokens=args.max_new_tokens, prompt_shots=prompt_shots, max_seq_length=args.max_seq_length).to(model.device)

            
            input_len = len(input_ids[0])
            # print('Input token length:', len(input_ids[0]))
            # print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)

            output_token = torch.tensor([], dtype=torch.long).to(device)

            past_key_values = None

            print(input_ids.shape)

            # inference_count = 0
            # accept_lengths = []
            with torch.inference_mode():

                cur_length = input_len

                step = 0
                
                for _ in range(args.max_new_tokens):

                    if step >= args.max_new_tokens:
                        break
                    
                    outputs = model(input_ids, past_key_values=past_key_values, return_dict=True, use_cache=True)
                    logits = outputs['logits']

                    next_token_logits = logits[:, -1:]
                    
                    next_token = torch.argmax(next_token_logits, dim=-1)

                    input_ids = next_token
                    past_key_values = outputs['past_key_values']

                    output_token = torch.cat([output_token, next_token], dim=-1)

                    if tokenizer.eos_token_id in next_token or cur_length + 1 >= args.max_seq_length:
                        break

                    cur_length += 1

                    step += 1

                # print(output_token)
            print('Token num:', step)
            print(f'Final output: {tokenizer.batch_decode(output_token, skip_special_tokens=True)}')

            result = tokenizer.batch_decode(output_token, skip_special_tokens=True)[0]

            print(result)

            rouge=rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

            if task_name == 'xsum':
                references = x['summary']
            elif task_name =='cnndm':
                references = x['highlights']

            clip_pred = result.find("\nArticle:")
            if clip_pred > 0:
                prediction = result[:clip_pred]
            else:
                prediction = result
            rouge_score = rouge.score(prediction, references)

            rouge_score = rouge_score['rouge2'].fmeasure

            all_rouge_score.append(rouge_score)

    print('Total time:', end_time - start_time)
    print('Avg rouge score:', np.mean(all_rouge_score))