from bayes_opt import BayesianOptimization
import random
import json

from decoding import infer, infer_input_ids
import json

from torch import nn
import torch.nn.functional as F

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # define GPU id, remove if you want to use all GPUs available
import torch
from contextlib import contextmanager
import numpy as np

# from transformers.models.llama.modeling_llama import LlamaForCausalLM 
from kv_cache import *
from utils import *

from copy import deepcopy
import matplotlib.pyplot as plt

from transformers import top_k_top_p_filtering
from transformers import AutoTokenizer

from datasets import load_dataset

from decoding import clip_input

import time

import random

from tqdm import tqdm

class LayerGroupSearching_bayesian_thru:
    def __init__(
        self,
        model,
        tokenizer,
        evaluate_prompts,
        prompt_shots,
        task_name,
        max_new_tokens,
        max_seq_length,
        bayesian_itr,
        num_groups=None,
        group_size_min=None,
        group_size_max=None,
        ts_layer_level=False,
        top_layers_level=False,
        top_layers_len=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluate_prompts = evaluate_prompts
        self.prompt_shots = prompt_shots
        self.task_name = task_name
        self.max_seq_length = max_seq_length
    
        self.num_all_layers = self.model.config.num_hidden_layers
        self.num_groups = num_groups
        self.group_size_min = group_size_min
        self.group_size_max = group_size_max
        self.max_new_tokens = max_new_tokens
        self.bayesian_itr = bayesian_itr
        self.ts_layer_level = ts_layer_level
        self.top_layers_level = top_layers_level
        self.top_layers_len = top_layers_len
       
    # 定义贝叶斯优化目标函数
    def obj_function(self, **params):
        
        if self.ts_layer_level:
            if self.top_layers_level:
                ts_params = [round(params[f'ts_{i}']) for i in range(self.top_layers_len, self.num_all_layers)]
            else:
                ts_params = [round(params[f'ts_{i}']) for i in range(self.num_all_layers)]
        elif self.top_layers_level:
            attn_params = [round(params[f'attn_{i}']) for i in range(self.top_layers_len, self.num_all_layers)]
            mlp_params = [round(params[f'mlp_{i}']) for i in range(self.top_layers_len, self.num_all_layers)]
        else:
            # 从 params 中提取 attn 和 mlp 的组号分配
            attn_params = [round(params[f'attn_{i}']) for i in range(self.num_all_layers)]
            mlp_params = [round(params[f'mlp_{i}']) for i in range(self.num_all_layers)]

        
        # 将参数转化为层分配方案
        attn_groups = [[] for _ in range(self.num_groups)]
        mlp_groups = [[] for _ in range(self.num_groups)]
        
        # print(len(attn_groups), len(mlp_groups))

        if self.ts_layer_level:

            if self.top_layers_level:
                for i in range(self.num_all_layers - self.top_layers_len):
                    attn_groups[ts_params[i]].append(i + self.top_layers_len)
                    mlp_groups[ts_params[i]].append(i + self.top_layers_len)
            else:
                for i in range(self.num_all_layers):
                    attn_groups[ts_params[i]].append(i)
                    mlp_groups[ts_params[i]].append(i)

        elif self.top_layers_level:
            for i in range(self.num_all_layers - self.top_layers_len):
                attn_groups[attn_params[i]].append(i + self.top_layers_len)
                mlp_groups[mlp_params[i]].append(i + self.top_layers_len)
                
            # print(attn_groups)
        else:
            for i in range(self.num_all_layers):
                attn_groups[attn_params[i]].append(i)
                mlp_groups[mlp_params[i]].append(i)

        # print(attn_groups)
        # print(mlp_groups)

        # # 去除空组
        attn_groups = [group for group in attn_groups if group]
        mlp_groups = [group for group in mlp_groups if group]

        # 确保每个组至少有 group_size_min 个层, 至多有 group_size_max 个层
        # for attn, mlp in zip(attn_groups, mlp_groups):
        #     if attn is None and mlp is None:
        #         return -1e2  # 返回一个极差的值，表示不可行解

        # for group in mlp_groups:
        #     if len(group) < self.group_size_min or len(group) > self.group_size_max:
        #         return -1e2  # 返回一个极差的值，表示不可行解
        
        # 调用原始的 evaluate_function
        objective_value = self.evaluate_function(
            self.model,
            self.tokenizer,
            self.evaluate_prompts,
            self.max_new_tokens,
            attn_groups, 
            mlp_groups,
            self.top_layers_len
        )
        return objective_value  # 贝叶斯优化默认是最大化


    def optimize_attn_mlp(
            self,
    ):

        # 定义贝叶斯优化的参数范围
        if self.ts_layer_level:
            if self.top_layers_level:
                pbounds = {
                    f'ts_{i}': (0, self.num_groups - 1) for i in range(self.top_layers_len, self.num_all_layers)
                }
            else:
                pbounds = {
                    f'ts_{i}': (0, self.num_groups - 1) for i in range(self.num_all_layers)
                }
        elif self.top_layers_level:
            pbounds = {
                f'attn_{i}': (0, self.num_groups - 1) for i in range(self.top_layers_len, self.num_all_layers)
            }
            pbounds.update({
                f'mlp_{i}': (0, self.num_groups - 1) for i in range(self.top_layers_len, self.num_all_layers)
            })
        else:
            pbounds = {
                f'attn_{i}': (0, self.num_groups - 1) for i in range(self.num_all_layers)
            }
            pbounds.update({
                f'mlp_{i}': (0, self.num_groups - 1) for i in range(self.num_all_layers)
            })

        # 使用贝叶斯优化器进行优化
        optimizer = BayesianOptimization(
            f=self.obj_function,
            pbounds=pbounds,
            random_state=42,
        )

        optimizer.set_gp_params(alpha=1e-2)


        # if self.top_layers_level:
        #     params = {}

        #     probe_attn_group =  [0, 1, 2, 0, 0, 1, 0, 2, 1, 1, 2, 1]
        #     probe_mlp_group = [1, 0, 2, 2, 0, 2, 2, 1, 2, 2, 1, 0]
            
        #     idx = 0
        #     for attn, mlp in zip(probe_attn_group, probe_mlp_group):
        #         params[f'attn_{idx + self.top_layers_len}'] = attn
        #         params[f'mlp_{idx + self.top_layers_len}'] = mlp
        #         idx += 1

        #     self.probe_attn_mlp(optimizer, params)

        # 执行优化
        optimizer.maximize(
            init_points=0,  # 初始随机探索次数
            n_iter=self.bayesian_itr,       # 优化迭代次数
        )


        # 输出最优解
        best_solution = optimizer.max
        if self.ts_layer_level:
            if self.top_layers_level:
                ts_solution = [int(best_solution['params'][f'ts_{i}']) for i in range(self.top_layers_len, self.num_all_layers)]

                # 将最优解保存为 JSON 文件
                solution_dict = {
                    "attn_solution": {f"Group_{j + 1}": [i for i in range(self.top_layers_len)] + [i + self.top_layers_len for i in range(self.num_all_layers - self.top_layers_len) if ts_solution[i] == j] for j in range(self.num_groups)},
                    "mlp_solution": {f"Group_{j + 1}": [i for i in range(self.top_layers_len)] + [i + self.top_layers_len for i in range(self.num_all_layers - self.top_layers_len) if ts_solution[i] == j] for j in range(self.num_groups)}
                }
            else:
                ts_solution = [int(best_solution['params'][f'ts_{i}']) for i in range(self.num_all_layers)]

                # 将最优解保存为 JSON 文件
                solution_dict = {
                    "attn_solution": {f"Group_{j + 1}": [i for i in range(self.num_all_layers) if ts_solution[i] == j] for j in range(self.num_groups)},
                    "mlp_solution": {f"Group_{j + 1}": [i for i in range(self.num_all_layers) if ts_solution[i] == j] for j in range(self.num_groups)}
                }

        elif self.top_layers_level:
            attn_solution = [int(best_solution['params'][f'attn_{i}']) for i in range(self.top_layers_len, self.num_all_layers)]
            mlp_solution = [int(best_solution['params'][f'mlp_{i}']) for i in range(self.top_layers_len, self.num_all_layers)]

            # 将最优解保存为 JSON 文件
            solution_dict = {
                "attn_solution": {f"Group_{j + 1}": [i for i in range(self.top_layers_len)] + [i + self.top_layers_len for i in range(self.num_all_layers - self.top_layers_len) if attn_solution[i] == j] for j in range(self.num_groups)},
                "mlp_solution": {f"Group_{j + 1}": [i for i in range(self.top_layers_len)] + [i + self.top_layers_len for i in range(self.num_all_layers - self.top_layers_len) if mlp_solution[i] == j] for j in range(self.num_groups)}
            }
        else:
            attn_solution = [int(best_solution['params'][f'attn_{i}']) for i in range(self.num_all_layers)]
            mlp_solution = [int(best_solution['params'][f'mlp_{i}']) for i in range(self.num_all_layers)]

            # 将最优解保存为 JSON 文件
            solution_dict = {
                "attn_solution": {f"Group_{j + 1}": [i for i in range(self.num_all_layers) if attn_solution[i] == j] for j in range(self.num_groups)},
                "mlp_solution": {f"Group_{j + 1}": [i for i in range(self.num_all_layers) if mlp_solution[i] == j] for j in range(self.num_groups)}
            }
        if self.ts_layer_level:

            with open(f'./ts_layer_level/optimal_solution_group_{self.num_groups}_itr_{self.bayesian_itr}_sample_{len(self.evaluate_prompts)}_gen_{self.max_new_tokens}_top_{self.top_layers_len}_new.json', 'w') as json_file:
                json.dump(solution_dict, json_file, indent=4)

        elif self.top_layers_level:
            with open(f'./attn_mlp_level/optimal_solution_group_{self.num_groups}_itr_{self.bayesian_itr}_sample_{len(self.evaluate_prompts)}_gen_{self.max_new_tokens}_top_{self.top_layers_len}.json', 'w') as json_file:
                json.dump(solution_dict, json_file, indent=4)

    def probe_attn_mlp(
            self, optimizer, params
    ):
        
        optimizer.probe(params=params, lazy=True)

        


    def evaluate_function(self, model, tokenizer, prompts, max_new_tokens, attn, mlp, top_layers_len):

        draft_attn_skip_masks = attn
        draft_mlp_skip_masks = mlp

        print(draft_attn_skip_masks)
        print(draft_mlp_skip_masks)

        total_time = 0
        total_tokens = 0

        for prompt in tqdm(prompts, desc="Optimizing attn and mlp"):
            # input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
            input_ids = clip_input(tokenizer, prompt, self.task_name, max_new_tokens, prompt_shots=self.prompt_shots, max_seq_length=self.max_seq_length).cuda()

            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model)
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

            model.current_length_data.zero_() # this is for rerun

            model.itr_count = 0
            # print(model.itr_count)
            
            input_len = len(input_ids[0])
            # print('Input token length:', len(input_ids[0]))
            # print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)

            inference_count = 0
            accept_lengths = []
            with torch.inference_mode():

                start_time = time.time()

                with model.self_draft():
                    # print(draft_attn_skip_masks)
                    draft_logits, base_outputs, base_logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values, draft_attn_skip_masks=draft_attn_skip_masks, draft_mlp_skip_masks=draft_mlp_skip_masks, top_layers_len=top_layers_len)
                inference_count += 1

                # print(draft_logits.shape) # (num_draft_layers, batch_size, seq_len, vocab_size)

                
                draft_pred = torch.argmax(draft_logits[..., -1, :], dim = -1)
                
                pred = torch.argmax(base_logits[..., -1, :], dim = -1)
                
                preds = torch.cat([pred, draft_pred[:, 0 ]], dim = -1)
                # print(preds.shape)

                # print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred)}')

                cur_length = input_len
                accept_lengths.append(1)
                for _ in range(max_new_tokens):

                    with model.self_draft():
                        draft_logits, base_outputs, base_logits = model(preds.cuda().unsqueeze(0), output_orig = True, past_key_values = model.past_key_values, draft_attn_skip_masks=draft_attn_skip_masks, draft_mlp_skip_masks=draft_mlp_skip_masks, top_layers_len=top_layers_len)
                    inference_count += 1
                    
                    draft_pred = torch.argmax(draft_logits[..., (-len(draft_attn_skip_masks)-1):, :], dim = -1)
                
                    pred = torch.argmax(base_logits[..., :, :], dim = -1)

                    posterior_mask = (
                                preds[1:] == pred[0, :-1]
                            ).int()
                    
                    accept_length = torch.cumprod(posterior_mask, dim = -1).sum().item()
                    
                    cur_length = cur_length + accept_length + 1
                    # update kv cache
                    model.current_length_data.fill_(cur_length)
                    # create new input
                    preds = torch.cat([pred[:, accept_length], draft_pred[:,0,accept_length]], dim = -1)
                    # preds = torch.cat([pred[:, accept_length], draft_pred[:accept_length,0,0]], dim = -1)
                    # print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred[0, :accept_length + 1])}')
                    accept_lengths.append(accept_length + 1)
                    if tokenizer.eos_token_id in pred[0, :accept_length + 1] or cur_length + draft_pred.shape[0] >= self.max_seq_length:
                        break
                
                end_time = time.time()

                duration = end_time - start_time

                total_time += duration

                total_tokens += max_new_tokens

        # time_per_token = duration / sum(accept_lengths)
        throughput = total_tokens / total_time

        return throughput