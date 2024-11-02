from decoding import infer, infer_input_ids
import json

import gurobipy as gp
from gurobipy import GRB

from torch import nn
import torch.nn.functional as F
from ortools.linear_solver import pywraplp

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

class LayerGroupSearching_thru:
    def __init__(
        self,
        model,
        tokenizer,
        evaluate_prompts,
        prompt_shots,
        task_name,
        max_new_tokens,
        num_groups=None,
        group_size_min=None,
        group_size_max=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.evaluate_prompts = evaluate_prompts
        self.prompt_shots = prompt_shots
        self.task_name = task_name
        
        self.gp_model = gp.Model("LayerGrouping")
        self.num_all_layers = self.model.config.num_hidden_layers
        self.num_groups = num_groups
        self.group_size_min = group_size_min
        self.group_size_max = group_size_max
        self.max_new_tokens = max_new_tokens

    def optimize_attn_mlp(
            self,
    ):
        
        # 初始化变量来存储所有输入的最优解
        global_optimal_attn_solution = None
        global_optimal_mlp_solution = None
        global_optimal_obj_value = float('inf')
        
        # 创建决策变量 x[i, j]，表示层 i 是否被分配到组 j
        attn = self.gp_model.addVars(self.num_all_layers, self.num_groups, vtype=GRB.BINARY, name="x")
        mlp = self.gp_model.addVars(self.num_all_layers, self.num_groups, vtype=GRB.BINARY, name="y")

        self.gp_model.update()

        # 随机初始化 attn 和 mlp 的初始值
        # for i in range(self.num_all_layers):
        #     for j in range(self.num_groups):
        #         attn[i, j].start = random.randint(0, 1)
        #         mlp[i, j].start = random.randint(0, 1)

        def distribute_layers(layers, num_groups, group_size_min, group_size_max):
            groups = [[] for _ in range(num_groups)]
            for g in range(num_groups):
                for _ in range(group_size_min):
                    if layers:  # Ensure layers are available to avoid pop() error
                        groups[g].append(layers.pop())
            for layer in layers:
                eligible_groups = [g for g in range(num_groups) if len(groups[g]) < group_size_max]
                if eligible_groups:  # Ensure eligible groups are available
                    selected_group = random.choice(eligible_groups)
                    groups[selected_group].append(layer)
            return groups

        # 随机打乱所有层的顺序
        layers_attn = list(range(self.num_all_layers))
        layers_mlp = list(range(self.num_all_layers))
        random.shuffle(layers_attn)
        random.shuffle(layers_mlp)

        # 分配 attn 和 mlp 的层到组
        groups_attn = distribute_layers(layers_attn, self.num_groups, self.group_size_min, self.group_size_max)
        groups_mlp = distribute_layers(layers_mlp, self.num_groups, self.group_size_min, self.group_size_max)

        print(groups_attn)
        print(groups_mlp)

        # 初始化 attn 的 start 值，按照分组结果设置
        for g, group_layers in enumerate(groups_attn):
            for i in range(self.num_all_layers):
                attn[i, g].start = 1 if i in group_layers else 0

        # 初始化 mlp 的 start 值，按照分组结果设置
        for g, group_layers in enumerate(groups_mlp):
            for i in range(self.num_all_layers):
                mlp[i, g].start = 1 if i in group_layers else 0
                
        self.gp_model.update()

        # 约束条件 1：每一层可以不被分配，如果被分配只能分配到一个组
        for i in range(self.num_all_layers):
            self.gp_model.addConstr(gp.quicksum(attn[i, j] for j in range(self.num_groups)) <= 1, name=f"attn_assign_{i}")
            self.gp_model.addConstr(gp.quicksum(mlp[i, j] for j in range(self.num_groups)) <= 1, name=f"mlp_assign_{i}")

        # 约束条件 2：每个组的元素数量在 L_min 和 L_max 之间
        for j in range(self.num_groups):
            self.gp_model.addConstr(gp.quicksum(attn[i, j] for i in range(self.num_all_layers)) >= self.group_size_min, name=f"attn_group_min_{j}")
            self.gp_model.addConstr(gp.quicksum(attn[i, j] for i in range(self.num_all_layers)) <= self.group_size_max, name=f"attn_group_max_{j}")

            self.gp_model.addConstr(gp.quicksum(mlp[i, j] for i in range(self.num_all_layers)) >= self.group_size_min, name=f"mlp_group_min_{j}")
            self.gp_model.addConstr(gp.quicksum(mlp[i, j] for i in range(self.num_all_layers)) <= self.group_size_max, name=f"mlp_group_max_{j}")

        # # 目标：最小化每个组的输出特征分布与原模型输出特征分布之间的差异
        # obj = gp.quicksum(self.compute_loss_attn_mlp([attn[i, j] for i in range(self.num_all_layers)], [mlp[i, j] for i in range(self.num_all_layers)], base_ret, OPT_STATUS) for j in range(self.num_groups))
        # # obj = gp.quicksum((y[j] - base_ret['logits']) ** 2 for j in range(self.num_groups))
        # self.gp_model.setObjective(obj, GRB.MINIMIZE)

        # 定义多个目标函数，每个组的差异都尽量小
        object = self.evaluate_function(
                    self.model,
                    self.tokenizer,
                    self.evaluate_prompts,
                    self.max_new_tokens,
                    attn, 
                    mlp, 
                ) 

        # 使用权重法来将多个目标合并为一个目标进行优化，这里使用等权重
        self.gp_model.setObjective(object, GRB.MAXIMIZE)

        # 求解模型
        self.gp_model.optimize()

        # 如果找到可行解，更新约束条件 3 和全局最优解
        if self.gp_model.status == GRB.OPTIMAL: 
    
            # 更新全局最优解
            if self.gp_model.ObjVal < global_optimal_obj_value:
                global_optimal_obj_value = self.gp_model.ObjVal
                global_optimal_attn_solution = [(i, j) for i in range(self.num_all_layers) for j in range(self.num_groups) if attn[i, j].X > 0.5]
                global_optimal_mlp_solution = [(i, j) for i in range(self.num_all_layers) for j in range(self.num_groups) if mlp[i, j].X > 0.5]
        
        else:
            print("No optimal solution found.")

        # 输出综合的最优解
        if global_optimal_attn_solution is not None and global_optimal_mlp_solution is not None:
            print("Global Optimal solution found:")

            # 创建包含 attn 和 mlp 最优解的字典
            solution_dict = {
                "attn_solution": {},
                "mlp_solution": {}
            }

            for j in range(self.num_groups):
                attn_group_layers = [i for i, group in global_optimal_attn_solution if group == j]
                print(f"attn Group {j + 1}: Layers {attn_group_layers}")
                solution_dict["attn_solution"][f"Group_{j + 1}"] = attn_group_layers

                mlp_group_layers = [i for i, group in global_optimal_mlp_solution if group == j]
                print(f"mlp Group {j + 1}: Layers {mlp_group_layers}")
                solution_dict["mlp_solution"][f"Group_{j + 1}"] = mlp_group_layers

                # 将解写入 JSON 文件
                with open('../optimal_solution.json', 'w') as json_file:
                    json.dump(solution_dict, json_file, indent=4)

        else:
            print("No optimal solution found.")


    def evaluate_function(self, model, tokenizer, prompts, max_new_tokens, attn, mlp):

        draft_attn_skip_masks = []
        draft_mlp_skip_masks = []

        for j in range(self.num_groups):
            draft_attn_skip_masks.append([i for i in range(self.num_all_layers) if attn[i, j].start > 0.5])
            draft_mlp_skip_masks.append([i for i in range(self.num_all_layers) if mlp[i, j].start > 0.5])

        total_time = 0
        total_tokens = 0

        for prompt in tqdm(prompts, desc="Optimizing attn and mlp"):
            # input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
            input_ids = clip_input(tokenizer, prompt, self.task_name, max_new_tokens, prompt_shots=self.prompt_shots).cuda()

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
                    draft_logits, base_outputs, base_logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values, draft_attn_skip_masks=draft_attn_skip_masks, draft_mlp_skip_masks=draft_mlp_skip_masks)
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
                        draft_logits, base_outputs, base_logits = model(preds.cuda().unsqueeze(0), output_orig = True, past_key_values = model.past_key_values, draft_attn_skip_masks=draft_attn_skip_masks, draft_mlp_skip_masks=draft_mlp_skip_masks)
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
                    if tokenizer.eos_token_id in pred[0, :accept_length + 1]:
                        break
                
                end_time = time.time()

                duration = end_time - start_time

                total_time += duration

                total_tokens += max_new_tokens

        # time_per_token = duration / sum(accept_lengths)
        throughput = total_tokens / total_time

        print(throughput)

        return throughput

        


