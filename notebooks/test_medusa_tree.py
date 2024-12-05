import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # define GPU id, remove if you want to use all GPUs available
import torch
from contextlib import contextmanager
import numpy as np
# from medusa.model.modeling_llama_ssd_v1 import LlamaForCausalLM
# from medusa.model.modeling_llama_ssd_v1_top_layers import LlamaForCausalLM
# from medusa.model.modeling_llama_ssd_v3 import LlamaForCausalLM
# from transformers.models.llama.modeling_llama import LlamaForCausalLM 
from medusa.model.medusa_model import MedusaModel
from medusa.model.medusa_choices import *
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

import time

from tqdm import tqdm

from rouge_score import rouge_scorer

# from utils import *

def clip_input(tokenizer, prompt, task_name, max_new_tokens=512, prompt_shots='', max_seq_length=2048):
    # print(prompt)
    if task_name == 'xsum':
        input_ids = tokenizer(
            prompt_shots +'Article: ' + prompt['document'] + '\nSummary:',
            return_tensors='pt').input_ids
    elif task_name == 'cnndm':
        input_ids = tokenizer(
            prompt_shots +'Article: ' + prompt['article'] + '\nSummary:',
            return_tensors='pt').input_ids
    elif task_name == 'humaneval':
        format_tabs=True
        if format_tabs:
            prompt = prompt['prompt'].replace("    ", "\t")
        else:
            prompt = prompt['prompt']
        input_ids = tokenizer(prompt,return_tensors='pt').input_ids
    if len(input_ids[0])+max_new_tokens>=max_seq_length:
        print(f'(input ids+max token)>max_seq_length {max_seq_length}')
        sample_num = (len(input_ids[0])+max_new_tokens-max_seq_length) 
        input_ids = torch.cat((input_ids[0][:2],input_ids[0][2:-3][:-sample_num],input_ids[0][-3:]),dim=0).unsqueeze(0)
    return  input_ids

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=False, help="Model name or path.", default='/root/MODELS/medusa-vicuna-7b-v1.3')
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
    parser.add_argument("--num_group", type=int, required=False, help="Draft group num.", default=3)
    parser.add_argument("--num_sample", type=int, required=False, help="Num sample.", default=10)
    parser.add_argument("--top_layers_len", type=int, required=False, help="top layers to keep", default=12)
    
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
        data = load_dataset('/home/zmw/xsum', split='test').shuffle(seed=seed).select(range(1000))
        shots = load_dataset('/home/zmw/xsum',split='train').shuffle(seed=seed).select(range(n_shot))
        prompt_keys=['document','summary']
    elif task_name == 'cnndm':
        data = load_dataset('/home/zmw/cnn_dailymail', name='3.0.0', split='test') .shuffle(seed=seed).select(range(1000))
        shots = load_dataset('/home/zmw/cnn_dailymail', name='3.0.0', split='train').shuffle(seed=seed).select(range(n_shot))
        prompt_keys=['article','highlights']
    for i in range(n_shot):
        prompt = 'Article: ' + shots[i][prompt_keys[0]] + '\nSummary: ' + shots[i][prompt_keys[1]].replace('\n', '') + '\n'
        prompt_shots += prompt


    model_name = args.model
    
    # config = LlamaConfig.from_pretrained(model_name)
    
    # print(config.num_skipped_draft_model)

    model = MedusaModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model = model.to(device)
    tokenizer = model.get_tokenizer()

    print(model.medusa_head)

    medusa_choices = mc_sim_7b_63

    total_avg_accept_length = 0

    start_time = time.time()

    all_rouge_score = []

    for idx,x in tqdm(enumerate(data)):

        if idx == args.num_sample:
            end_time = time.time()
            break

        input_ids = clip_input(tokenizer, x, task_name, max_new_tokens=args.max_new_tokens, prompt_shots=prompt_shots, max_seq_length=args.max_seq_length).to(model.device)

        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

        model.current_length_data.zero_() # this is for rerun
        
        input_len = len(input_ids[0])
        # print('Input token length:', len(input_ids[0]))
        # print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)

        accept_lengths_tree = []
        with torch.inference_mode():

            new_token = 0

            reset_medusa_mode(model)
            medusa_buffers = generate_medusa_buffers(
                        medusa_choices, device=model.base_model.device
                    )
    
            medusa_logits, logits = initialize_medusa(
                input_ids, model, medusa_buffers["medusa_attn_mask"], past_key_values
            )

            cur_length = input_len + 1
            accept_lengths_tree.append(1)
            step = 0
            for _ in range(args.max_new_tokens):

                if step >= args.max_new_tokens:
                    break

                candidates, tree_candidates = generate_candidates(
                        medusa_logits,
                        logits,
                        medusa_buffers["tree_indices"],
                        medusa_buffers["retrieve_indices"],
                    )
                medusa_logits, logits, outputs = tree_decoding(
                        model,
                        tree_candidates,
                        past_key_values,
                        medusa_buffers["medusa_position_ids"],
                        input_ids,
                        medusa_buffers["retrieve_indices"],
                    )
                best_candidate, accept_length = evaluate_posterior(
                        logits, candidates, temperature = 0, posterior_threshold = 0, posterior_alpha = 0
                    )
                input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                        input_ids,
                        candidates,
                        best_candidate,
                        accept_length,
                        medusa_buffers["retrieve_indices"],
                        outputs,
                        logits,
                        medusa_logits,
                        new_token,
                        past_key_values_data,
                        current_length_data,
                    )
                
                accept_length_tree = input_ids.shape[1] - cur_length
                cur_length = accept_length_tree + cur_length
                accept_lengths_tree.append(accept_length_tree)
                step += accept_length_tree
                if model.tokenizer.eos_token_id in input_ids[0, input_len:] or cur_length + new_token >= args.max_seq_length:
                    break
        
        print('Decode:', tokenizer.batch_decode(input_ids[:,input_len:]))  

        result = tokenizer.batch_decode(input_ids[:,input_len:])[0]

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
        # plt.plot(accept_lengths)
        # plt.xlabel('Inference step')
        # plt.ylabel('Accept length')
        # plt.savefig('accept_length.png')
        print('Avg. accept length:', np.mean(accept_lengths_tree))
        print('Token num:', step)

        total_avg_accept_length += np.mean(accept_lengths_tree)
    
    print('Total avg. accept length:', total_avg_accept_length / args.num_sample)
    print('Total time:', end_time - start_time)
    print('Avg rouge score:', np.mean(all_rouge_score))