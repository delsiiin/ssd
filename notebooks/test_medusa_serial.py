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
    # print(model.medusa)

    # for name, params in model.medusa_head[0].named_parameters():
    #     print(name, params)

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

        output_token = torch.tensor([], dtype=torch.long).to(device)

        inference_count = 0
        accept_lengths = []
        with torch.inference_mode():
    
            medusa_logits, outputs, logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values, medusa_forward=True)
            inference_count += 1

            medusa_pred = torch.argmax(medusa_logits[..., -1, :], dim = -1)
            pred = torch.argmax(logits[..., -1, :], dim = -1)
            preds = torch.cat([pred, medusa_pred[:, 0 ]], dim = -1)
            # print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred)}')
            output_token = torch.cat([output_token, pred], dim = -1)

            cur_length = input_len
            accept_lengths.append(1)
            step = 0
            for _ in range(args.max_new_tokens):
                
                if step >= args.max_new_tokens:
                    break

                medusa_logits, outputs, logits = model(preds.cuda().unsqueeze(0), output_orig = True, past_key_values = model.past_key_values, medusa_forward=True)
                inference_count += 1

                medusa_pred = torch.argmax(medusa_logits[..., -6:, :], dim = -1)
                pred = torch.argmax(logits[..., :, :], dim = -1)
                posterior_mask = (
                            preds[1:] == pred[0, :-1]
                        ).int()
                accept_length = torch.cumprod(posterior_mask, dim = -1).sum().item()
                cur_length = cur_length + accept_length + 1
                # update kv cache
                model.current_length_data.fill_(cur_length)
                # create new input
                preds = torch.cat([pred[:, accept_length], medusa_pred[:,0,accept_length]], dim = -1)
                output_token = torch.cat([output_token, pred[0, :accept_length + 1]], dim = -1)
                # print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred[0, :accept_length + 1])}')
                accept_lengths.append(accept_length + 1)
                step += accept_length + 1
                if tokenizer.eos_token_id in pred[0, :accept_length + 1] or cur_length + medusa_pred.shape[0] >= args.max_seq_length:
                    break
        
        print(f'Final output: {tokenizer.decode(output_token, skip_special_tokens=True)}')

        result = tokenizer.decode(output_token, skip_special_tokens=True)

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
        print('Avg. accept length:', np.mean(accept_lengths))
        print('Token num:', step)

        total_avg_accept_length += np.mean(accept_lengths)
    
    print('Total avg. accept length:', total_avg_accept_length / args.num_sample)
    print('Total time:', end_time - start_time)
    print('Avg rouge score:', np.mean(all_rouge_score))