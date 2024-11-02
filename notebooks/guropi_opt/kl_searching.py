from decoding import infer, infer_input_ids
import json

import gurobipy as gp
from gurobipy import GRB

from torch import nn
import torch.nn.functional as F
from ortools.linear_solver import pywraplp

import random

from tqdm import tqdm

class LayerGroupSearching_KL:
    def __init__(
        self,
        model,
        tokenizer,
        evaluate_prompts,
        num_groups=None,
        group_size_min=None,
        group_size_max=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.evaluate_prompts = evaluate_prompts
        
        self.gp_model = gp.Model("LayerGrouping")
        self.num_all_layers = self.model.config.num_hidden_layers
        self.num_groups = num_groups
        self.group_size_min = group_size_min
        self.group_size_max = group_size_max

    def optimize_transformer_layer(
            self,
    ):
        
        # 初始化变量来存储所有输入的最优解
        global_optimal_solution = None
        global_optimal_obj_value = float('inf')

        for prompt in self.evaluate_prompts:

            OPT_STATUS = False

            base_ret = infer(self.model, self.tokenizer, prompt, generate_fn='base')

            # 创建决策变量 x[i, j]，表示层 i 是否被分配到组 j
            x = self.gp_model.addVars(self.num_all_layers, self.num_groups, vtype=GRB.BINARY, name="x")

            # 约束条件 1：每一层可以不被分配，如果被分配只能分配到一个组
            for i in range(self.num_all_layers):
                self.gp_model.addConstr(gp.quicksum(x[i, j] for j in range(self.num_groups)) <= 1, name=f"layer_assign_{i}")

            # 约束条件 2：每个组的元素数量在 L_min 和 L_max 之间
            for j in range(self.num_groups):
                self.gp_model.addConstr(gp.quicksum(x[i, j] for i in range(self.num_all_layers)) >= self.group_size_min, name=f"group_min_{j}")
                self.gp_model.addConstr(gp.quicksum(x[i, j] for i in range(self.num_all_layers)) <= self.group_size_max, name=f"group_max_{j}")

            # 目标：最小化每个组的输出特征分布与原模型输出特征分布之间的差异
            obj = gp.quicksum(self.compute_loss_transformer_layer([x[i, j] for i in range(self.num_all_layers)], base_ret, OPT_STATUS) for j in range(self.num_groups))
            # obj = gp.quicksum((y[j] - base_ret['logits']) ** 2 for j in range(self.num_groups))
            self.gp_model.setObjective(obj, GRB.MINIMIZE)

            
            # 求解模型
            self.gp_model.optimize()

            
            # 如果找到可行解，更新约束条件 3 和全局最优解
            if self.gp_model.status == GRB.OPTIMAL:
                
                OPT_STATUS = True
        
                # 更新全局最优解
                if self.gp_model.ObjVal < global_optimal_obj_value:
                    global_optimal_obj_value = self.gp_model.ObjVal
                    global_optimal_solution = [(i, j) for i in range(self.num_all_layers) for j in range(self.num_groups) if x[i, j].X > 0.5]
            
            else:
                print("No optimal solution found.")

        # 输出综合的最优解
        if global_optimal_solution is not None:
            print("Global Optimal solution found:")
            for j in range(self.num_groups):
                group_layers = [i for i, group in global_optimal_solution if group == j]
                print(f"Group {j + 1}: Layers {group_layers}")
        else:
            print("No optimal solution found.")


    def compute_loss_transformer_layer(self, x, base_output, status, temperature=2.0, alpha=0.5):

        group_input = base_output['input_ids'].clone()
        if status == "False":
            layer_group = [i for i in range(self.num_all_layers)]
        elif status == "Verify":
            layer_group = x
        elif status == "True":
            layer_group = [i for i in range(self.num_all_layers) if x[i].X > 0.5]
        group_output = infer_input_ids(self.model, self.tokenizer, group_input, generate_fn='ssd', attn_group=layer_group, mlp_group=layer_group)
        draft_logits = group_output['logits']
        draft_loss = group_output['loss']

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(draft_logits / temperature, dim=-1),
                F.softmax(base_output['logits'] / temperature, dim=-1),
            )
            * (temperature ** 2)
        )

        print(loss_logits)

        # Return weighted student loss
        # loss = alpha * draft_loss + (1.0 - alpha) * loss_logits
        return loss_logits.item()


    def optimize_attn_mlp(
            self,
    ):
        
        # 初始化变量来存储所有输入的最优解
        global_optimal_attn_solution = None
        global_optimal_mlp_solution = None
        global_optimal_obj_value = float('inf')
        optimal_obj_values = [float('inf')] * self.num_groups

        # for prompt in self.evaluate_prompts:

        #     OPT_STATUS = "False"

        #     base_ret = infer(self.model, self.tokenizer, prompt, generate_fn='base')

        #     # 创建决策变量 x[i, j]，表示层 i 是否被分配到组 j
        #     attn = self.gp_model.addVars(self.num_all_layers, self.num_groups, vtype=GRB.BINARY, name="x")
        #     mlp = self.gp_model.addVars(self.num_all_layers, self.num_groups, vtype=GRB.BINARY, name="y")

        #     # 约束条件 1：每一层可以不被分配，如果被分配只能分配到一个组
        #     for i in range(self.num_all_layers):
        #         self.gp_model.addConstr(gp.quicksum(attn[i, j] for j in range(self.num_groups)) <= 1, name=f"attn_assign_{i}")
        #         self.gp_model.addConstr(gp.quicksum(mlp[i, j] for j in range(self.num_groups)) <= 1, name=f"mlp_assign_{i}")

        #     # 约束条件 2：每个组的元素数量在 L_min 和 L_max 之间
        #     for j in range(self.num_groups):
        #         self.gp_model.addConstr(gp.quicksum(attn[i, j] for i in range(self.num_all_layers)) >= self.group_size_min, name=f"attn_group_min_{j}")
        #         self.gp_model.addConstr(gp.quicksum(attn[i, j] for i in range(self.num_all_layers)) <= self.group_size_max, name=f"attn_group_max_{j}")

        #         self.gp_model.addConstr(gp.quicksum(mlp[i, j] for i in range(self.num_all_layers)) >= self.group_size_min, name=f"mlp_group_min_{j}")
        #         self.gp_model.addConstr(gp.quicksum(mlp[i, j] for i in range(self.num_all_layers)) <= self.group_size_max, name=f"mlp_group_max_{j}")

        #     # # 目标：最小化每个组的输出特征分布与原模型输出特征分布之间的差异
        #     # obj = gp.quicksum(self.compute_loss_attn_mlp([attn[i, j] for i in range(self.num_all_layers)], [mlp[i, j] for i in range(self.num_all_layers)], base_ret, OPT_STATUS) for j in range(self.num_groups))
        #     # # obj = gp.quicksum((y[j] - base_ret['logits']) ** 2 for j in range(self.num_groups))
        #     # self.gp_model.setObjective(obj, GRB.MINIMIZE)

            
        #      # 定义多个目标函数，每个组的差异都尽量小
        #     objectives = []
        #     for j in range(self.num_groups):
        #         obj_j = self.compute_loss_attn_mlp([attn[i, j] for i in range(self.num_all_layers)], [mlp[i, j] for i in range(self.num_all_layers)], base_ret, OPT_STATUS)
        #         objectives.append(obj_j)

        #     # 使用权重法来将多个目标合并为一个目标进行优化，这里使用等权重
        #     combined_obj = gp.quicksum(objectives[j] for j in range(self.num_groups))
        #     self.gp_model.setObjective(combined_obj, GRB.MINIMIZE)


        #     # 求解模型
        #     self.gp_model.optimize()

        #     # 如果找到可行解，更新约束条件 3 和全局最优解
        #     if self.gp_model.status == GRB.OPTIMAL:
                
        #         OPT_STATUS = "True"
        
        #         # 更新全局最优解
        #         if self.gp_model.ObjVal < global_optimal_obj_value:
        #             global_optimal_obj_value = self.gp_model.ObjVal
        #             global_optimal_attn_solution = [(i, j) for i in range(self.num_all_layers) for j in range(self.num_groups) if attn[i, j].X > 0.5]
        #             global_optimal_mlp_solution = [(i, j) for i in range(self.num_all_layers) for j in range(self.num_groups) if mlp[i, j].X > 0.5]

        #             for j in range(self.num_groups):
        #                 optimal_obj_values[j] = objectives[j]
            
        #     else:
        #         print("No optimal solution found.")

        # 创建决策变量 x[i, j]，表示层 i 是否被分配到组 j
        attn = self.gp_model.addVars(self.num_all_layers, self.num_groups, vtype=GRB.BINARY, name="x")
        mlp = self.gp_model.addVars(self.num_all_layers, self.num_groups, vtype=GRB.BINARY, name="y")

        self.gp_model.update()

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
        combined_obj_all_prompts = gp.quicksum(
            gp.quicksum(
                self.compute_loss_attn_mlp(
                    attn, 
                    mlp, 
                    infer(self.model, self.tokenizer, prompt, generate_fn='base'), 
                    j
                ) 
                for j in tqdm(range(self.num_groups), desc="Groups")
            )
            for prompt in tqdm(self.evaluate_prompts, desc="Prompts")
        )

        # 使用权重法来将多个目标合并为一个目标进行优化，这里使用等权重
        combined_obj = combined_obj_all_prompts
        self.gp_model.setObjective(combined_obj, GRB.MINIMIZE)

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
            
    def compute_loss_attn_mlp(self, attn, mlp,  base_output, group_idx, temperature=2.0, alpha=0.5):

        group_input = base_output['input_ids'].clone()

        attn_group = [i for i in range(self.num_all_layers) if attn[i, group_idx].start > 0.5]
        mlp_group = [i for i in range(self.num_all_layers) if mlp[i, group_idx].start > 0.5]

        group_output = infer_input_ids(self.model, self.tokenizer, group_input, generate_fn='ssd', attn_group=attn_group, mlp_group=mlp_group)
        draft_logits = group_output['logits']
        draft_loss = group_output['loss']

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(draft_logits / temperature, dim=-1),
                F.softmax(base_output['logits'] / temperature, dim=-1),
            )
            * (temperature ** 2)
        )

        # print(loss_logits)

        # Return weighted student loss
        # loss = alpha * draft_loss + (1.0 - alpha) * loss_logits
        return loss_logits.item()
    

    def optimize_attn_mlp_middle(
            self,
            top_layers,
    ):
        
        self.num_all_layers = self.num_all_layers - len(top_layers)

        # 初始化变量来存储所有输入的最优解
        global_optimal_attn_solution = None
        global_optimal_mlp_solution = None
        global_optimal_obj_value = float('inf')
        optimal_obj_values = [float('inf')] * self.num_groups

        for prompt in self.evaluate_prompts:

            OPT_STATUS = "False"

            base_ret = infer(self.model, self.tokenizer, prompt, generate_fn='base')

            # 创建决策变量 x[i, j]，表示层 i 是否被分配到组 j
            attn = self.gp_model.addVars(self.num_all_layers, self.num_groups, vtype=GRB.BINARY, name="x")
            mlp = self.gp_model.addVars(self.num_all_layers, self.num_groups, vtype=GRB.BINARY, name="y")

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
            objectives = []
            for j in range(self.num_groups):
                obj_j = self.compute_loss_attn_mlp_middle([attn[i, j] for i in range(self.num_all_layers)], [mlp[i, j] for i in range(self.num_all_layers)], base_ret, OPT_STATUS, top_layers)
                objectives.append(obj_j)

            # 使用权重法来将多个目标合并为一个目标进行优化，这里使用等权重
            combined_obj = gp.quicksum(objectives[j] for j in range(self.num_groups))
            self.gp_model.setObjective(combined_obj, GRB.MINIMIZE)


            # 求解模型
            self.gp_model.optimize()

            # 如果找到可行解，更新约束条件 3 和全局最优解
            if self.gp_model.status == GRB.OPTIMAL:
                
                OPT_STATUS = "True"
        
                # 更新全局最优解
                if self.gp_model.ObjVal < global_optimal_obj_value:
                    global_optimal_obj_value = self.gp_model.ObjVal
                    global_optimal_attn_solution = [(i, j) for i in range(self.num_all_layers) for j in range(self.num_groups) if attn[i, j].X > 0.5]
                    global_optimal_mlp_solution = [(i, j) for i in range(self.num_all_layers) for j in range(self.num_groups) if mlp[i, j].X > 0.5]

                    for j in range(self.num_groups):
                        optimal_obj_values[j] = objectives[j]
            
            else:
                print("No optimal solution found.")

        # 输出综合的最优解
        if global_optimal_attn_solution is not None and global_optimal_mlp_solution is not None:
            print("Global Optimal solution found:")

            for j in range(self.num_groups):
                attn_group_layers = [i+len(top_layers) for i, group in global_optimal_attn_solution if group == j]
                print(f"attn Group {j + 1}: Layers {attn_group_layers}")

                mlp_group_layers = [i+len(top_layers) for i, group in global_optimal_mlp_solution if group == j]
                print(f"mlp Group {j + 1}: Layers {mlp_group_layers}")

                group_loss = self.compute_loss_attn_mlp_middle(attn_group_layers, mlp_group_layers, base_ret, "Verify", top_layers)
                print(f"Group {j + 1} Loss: {group_loss}")

        else:
            print("No optimal solution found.")
    

    def compute_loss_attn_mlp_middle(self, attn, mlp,  base_output, status, top, temperature=2.0, alpha=0.5):

        group_input = base_output['input_ids'].clone()
        if status == "False":
            attn_group = [i for i in range(self.num_all_layers)]
            mlp_group = [i for i in range(self.num_all_layers)]
        elif status == "True":
            attn_group = [i for i in range(self.num_all_layers) if attn[i].X > 0.5]
            mlp_group = [i for i in range(self.num_all_layers) if mlp[i].X > 0.5]

            attn_group = top + attn_group
            mlp_group = top + mlp_group
        elif status == "Verify":
            attn_group = attn
            mlp_group = mlp

            attn_group = top + attn_group
            mlp_group = top + mlp_group

        group_output = infer_input_ids(self.model, self.tokenizer, group_input, generate_fn='ssd', attn_group=attn_group, mlp_group=mlp_group)
        draft_logits = group_output['logits']
        draft_loss = group_output['loss']

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(draft_logits / temperature, dim=-1),
                F.softmax(base_output['logits'] / temperature, dim=-1),
            )
            * (temperature ** 2)
        )

        print(loss_logits)

        # Return weighted student loss
        # loss = alpha * draft_loss + (1.0 - alpha) * loss_logits
        return loss_logits.item()
