import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # define GPU id, remove if you want to use all GPUs available
import torch
from contextlib import contextmanager
import numpy as np
# from medusa.model.modeling_llama_ssd_v1 import LlamaForCausalLM
# from medusa.model.modeling_llama_ssd_v1_top_layers import LlamaForCausalLM
from medusa.model.modeling_llama_ssd_router import LlamaForCausalLM, add_router
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
    
    config_kwargs = {}

    config_kwargs["top_layers_len"] = args.top_layers_len 
    config_kwargs["top_k_group"] = args.top_k_group 
    config_kwargs["resnet_num"] = args.resnet_num 

    config = LlamaConfig.from_pretrained(model_name, **config_kwargs)
    
    # print(config.num_skipped_draft_model)

    lora_path = f"/root/idea/speculative_decoding/Medusa/axolotl/vicuna-7b-v1.3-qlora-ssd-out-router-top-{args.top_layers_len}-k-{args.top_k_group}-seq-2048"
    lora_config = PeftConfig.from_pretrained(lora_path)

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        config=config,
    )
    model = model.eval().to(device)

    add_router(model)

    model = PeftModel.from_pretrained(model, lora_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_avg_accept_length = 0

    start_time = time.time()

    with torch.no_grad():

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

            model.itr_count = 0
            
            input_len = len(input_ids[0])
            # print('Input token length:', len(input_ids[0]))
            # print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)

            output_token = torch.tensor([], dtype=torch.long).to(device)

            inference_count = 0
            accept_lengths = []
            with torch.inference_mode():
        
                with model.self_draft():
                    # print(draft_attn_skip_masks)
                    draft_logits, base_logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values, top_layers_len=args.top_layers_len)
                inference_count += 1

                # print(draft_logits.shape) # (num_draft_layers, batch_size, seq_len, vocab_size)

                ############### Use Sampling ################
                do_sample = False
                if do_sample:
                    # draft_pred = sample(draft_logits[..., -1, :], do_sample=do_sample, top_k=50, top_p=0.8, temperature=0.7)
                    # draft_pred = get_nucleus_one_token(draft_logits[..., -1, :].view(-1, draft_logits.shape[-1]), temperature=0.7, top_p=0.8)
                    draft_pred = get_typical_one_token(draft_logits[..., -1, :].view(-1, draft_logits.shape[-1]), temperature=args.temperature, posterior_threshold=args.posterior_threshold, posterior_alpha=args.posterior_alpha).view(draft_logits.shape[0], -1)
                else:
                    draft_pred = torch.argmax(draft_logits[..., -1, :], dim = -1)
                #############################################  

                ############### Use Sampling ################
                do_sample = False
                if do_sample:
                    # pred = sample(base_logits[..., -1, :], do_sample=do_sample, top_k=50, top_p=0.8, temperature=0.7)
                    # pred = get_nucleus_one_token(base_logits[..., -1, :].view(-1, base_logits.shape[-1]), temperature=0.7, top_p=0.8).view(-1)
                    pred = get_typical_one_token(base_logits[..., -1, :].view(-1, base_logits.shape[-1]), temperature=args.temperature, posterior_threshold=args.posterior_threshold, posterior_alpha=args.posterior_alpha).view(-1)
                else:
                    pred = torch.argmax(base_logits[..., -1, :], dim = -1)
                #############################################
                
                preds = torch.cat([pred, draft_pred[:, 0 ]], dim = -1)
                # print(preds.shape)

                # print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred)}')
                output_token = torch.cat([output_token, pred], dim = -1)

                cur_length = input_len
                accept_lengths.append(1)
                for _ in range(args.max_new_tokens):
                    with model.self_draft():
                        draft_logits, base_logits = model(preds.cuda().unsqueeze(0), output_orig = True, past_key_values = model.past_key_values, top_layers_len=args.top_layers_len)
                    inference_count += 1

                    ############### Use Sampling ################
                    do_sample = False
                    if do_sample:
                        # draft_pred = sample(draft_logits[..., -5:, :], do_sample=do_sample, top_k=50, top_p=0.8, temperature=0.7)
                        # draft_pred = get_nucleus_one_token(draft_logits[..., -5:, :].view(-1, draft_logits.shape[-1]), temperature=0.7, top_p=0.8)
                        draft_pred = get_typical_one_token(draft_logits[..., -5:, :].view(-1, draft_logits.shape[-1]), temperature=args.temperature, posterior_threshold=args.posterior_threshold, posterior_alpha=args.posterior_alpha).view(draft_logits.shape[0], -1, 5)
                    else:
                        draft_pred = torch.argmax(draft_logits[..., (-args.top_k_group-1):, :], dim = -1)
                    #############################################
                    # print(draft_pred.shape)

                    ############### Use Sampling ################
                    do_sample = False
                    if do_sample:
                        # pred = sample(base_logits[..., :, :], do_sample=do_sample, top_k=50, top_p=0.8, temperature=0.7)
                        # pred = get_nucleus_one_token(base_logits[..., :, :].view(-1, base_logits.shape[-1]), temperature=0.7, top_p=0.8).view(base_logits.shape[:-1])
                        pred = get_typical_one_token(base_logits[..., :, :].view(-1, base_logits.shape[-1]), temperature=args.temperature, posterior_threshold=args.posterior_threshold, posterior_alpha=args.posterior_alpha).view(base_logits.shape[:-1])
                    else:
                        pred = torch.argmax(base_logits[..., :, :], dim = -1)
                        # print(pred.shape)
                    #############################################

                    # print(f"base logits {base_logits.shape}, draft pred {draft_pred.shape}")

                    do_sample = False
                    if do_sample:
                        _, accept_length = evaluate_posterior(
                                base_logits, preds, temperature=args.temperature, posterior_threshold=args.posterior_threshold, posterior_alpha=args.posterior_alpha
                            )
                        accept_length = accept_length.cpu()
                    else:
                        posterior_mask = (
                                    preds[1:] == pred[0, :-1]
                                ).int()
                        accept_length = torch.cumprod(posterior_mask, dim = -1).sum().item()
                    
                    # print(accept_length)

                    cur_length = cur_length + accept_length + 1
                    # update kv cache
                    model.current_length_data.fill_(cur_length)
                    # create new input
                    preds = torch.cat([pred[:, accept_length], draft_pred[:,0,accept_length]], dim = -1)
                    output_token = torch.cat([output_token, pred[0, :accept_length + 1]], dim = -1)
                    # preds = torch.cat([pred[:, accept_length], draft_pred[:accept_length,0,0]], dim = -1)
                    # print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred[0, :accept_length + 1])}')
                    accept_lengths.append(accept_length + 1)
                    if tokenizer.eos_token_id in pred[0, :accept_length + 1] or cur_length + draft_pred.shape[0] >= args.max_seq_length:
                        break
                
            print(f'Final output: {tokenizer.decode(output_token, skip_special_tokens=True)}')

            # plt.plot(accept_lengths)
            # plt.xlabel('Inference step')
            # plt.ylabel('Accept length')
            # plt.savefig('accept_length.png')
            print('Avg. accept length:', np.mean(accept_lengths))
            print('Token num:', np.sum(accept_lengths))

            total_avg_accept_length += np.mean(accept_lengths)
    
    print('Total avg. accept length:', total_avg_accept_length / args.num_sample)
    print('Total time:', end_time - start_time)