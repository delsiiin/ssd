import gurobipy as gp
from gurobipy import GRB
import torch
from transformers import LlamaModel, LlamaConfig

# 假设已知的数据
m = 10  # 层的数量（Transformer层的数量）
n = 3   # 组的数量
L_min = 2  # 每组最少的层数
L_max = 5  # 每组最多的层数

# 使用Llama模型生成层的输出
config = LlamaConfig(num_hidden_layers=m)
model = LlamaModel(config)

# 随机生成输入张量，假设序列长度为16，隐藏维度与模型配置一致
input_tensor = torch.rand(1, 16, config.hidden_size)

# 获取每一层的输出特征（hidden_states）
x = input_tensor
for layer in model.layers:
    x = layer(x)[0]  # 获取每一层的输出特征
    
# 假设原始模型的输出特征分布为最终层的输出特征
y_orig = x.mean().item()

# 创建Gurobi模型
gurobi_model = gp.Model("LayerGrouping")

# 创建决策变量 x[i, j]，表示层 i 是否被分配到组 j
x = gurobi_model.addVars(m, n, vtype=GRB.BINARY, name="x")

# 创建目标函数中的辅助变量 y[j]，表示组 j 的输出特征分布
y = gurobi_model.addVars(n, vtype=GRB.CONTINUOUS, name="y")

# 约束条件 1：每一层必须且只能分配到一个组
for i in range(m):
    gurobi_model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"layer_assign_{i}")

# 约束条件 2：每个组的元素数量在 L_min 和 L_max 之间
for j in range(n):
    gurobi_model.addConstr(gp.quicksum(x[i, j] for i in range(m)) >= L_min, name=f"group_min_{j}")
    gurobi_model.addConstr(gp.quicksum(x[i, j] for i in range(m)) <= L_max, name=f"group_max_{j}")

# 约束条件 3：定义组的输出特征分布 y[j]
for j in range(n):
    # 组 j 的输出定义为输入通过组内每一个层后的输出
    group_output = input_tensor.clone()
    for i in range(m):
        print(x[i, j].X)
        if x[i, j].X > 0.5:  # 判断层是否属于组 j
            group_output = model.layers[i](group_output)[0]  # 获取通过该层的输出
    gurobi_model.addConstr(y[j] == group_output.mean().item(), name=f"group_output_{j}")

# 目标：最小化每个组的输出特征分布与原模型输出特征分布之间的差异
obj = gp.quicksum((y[j] - y_orig) * (y[j] - y_orig) for j in range(n))
gurobi_model.setObjective(obj, GRB.MINIMIZE)

# 求解模型
gurobi_model.optimize()

# 输出结果
if gurobi_model.status == GRB.OPTIMAL:
    print("Optimal solution found:\n")
    for j in range(n):
        group_layers = [i for i in range(m) if x[i, j].X > 0.5]
        print(f"Group {j + 1}: Layers {group_layers}")
        print(f"Group {j + 1} Output: {y[j].X}")
else:
    print("No optimal solution found.")
