import time

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # define GPU id, remove if you want to use all GPUs available
import torch
from contextlib import contextmanager
import numpy as np

from copy import deepcopy
import matplotlib.pyplot as plt

from transformers import top_k_top_p_filtering


def sample(logits, return_probs: bool=False, do_sample: bool=False, top_k: int=50, top_p: float=0.7, temperature: float=0.7):

    if return_probs:

        all_probs = logits.softmax(-1)
        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
            probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        else:
            probs, output_ids = torch.max(all_probs, dim=-1)
            
        return output_ids, probs

    else:

        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
        else:
            output_ids = torch.argmax(logits, dim=-1)
            
        return output_ids

def base_generate(model, tokenizer, input_ids):

    current_input_ids = input_ids
    
    with torch.no_grad():
        output = model(input_ids=current_input_ids,
                return_dict=True,
                use_cache=False)
        logits = output.logits
                
    return {
        'logits': logits,
    }

def self_speculative_generate(model, tokenizer, input_ids, attn_group, mlp_group):

    with torch.no_grad():
        with model.self_draft():
            draft_output = model(input_ids=input_ids,
                return_dict=True,
                use_cache=False,
                draft_attn_skip_mask=attn_group,
                draft_mlp_skip_mask=mlp_group)
            draft_logits = draft_output.logits
            draft_loss = draft_output.loss
            
    return {
        'logits': draft_logits,
        'loss': draft_loss,
    }


generate_fn_mapping = {
    'base': base_generate,
    'ssd':  self_speculative_generate,
}

def infer(model, tokenizer, prompt, generate_fn='base', 
          decode_timing=True, seed=42, *args, **kargs):

    if isinstance(generate_fn, str):
        generate_fn = generate_fn_mapping[generate_fn]

    if seed is not None:
        torch.manual_seed(seed)
              
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    if decode_timing:
        tic = time.time()
    generate_dict = generate_fn(model, tokenizer, input_ids, *args, **kargs)
    if decode_timing:
        toc = time.time()
        decode_time = toc - tic
    else:
        decode_time = None
    generate_dict['time'] = decode_time
    generate_dict['input_ids'] = input_ids
    return generate_dict

def infer_input_ids(model, tokenizer, input_ids, generate_fn='base', 
          decode_timing=True, seed=42, *args, **kargs):

    if isinstance(generate_fn, str):
        generate_fn = generate_fn_mapping[generate_fn]

    if seed is not None:
        torch.manual_seed(seed)
              
    input_ids = input_ids.to(model.device)
    if decode_timing:
        tic = time.time()
    generate_dict = generate_fn(model, tokenizer, input_ids, *args, **kargs)
    if decode_timing:
        toc = time.time()
        decode_time = toc - tic
    else:
        decode_time = None
    generate_dict['time'] = decode_time
    generate_dict['input_ids'] = input_ids
    return generate_dict


def clip_input(tokenizer, prompt, task_name, max_new_tokens=512, prompt_shots='', max_seq_length=2048):
    if task_name == 'xsum':
        input_ids = tokenizer(
            prompt_shots +'Article: ' + prompt['document'] + '\nSummary:',
            return_tensors='pt').input_ids
    elif task_name == 'cnn_dailymail':
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
    