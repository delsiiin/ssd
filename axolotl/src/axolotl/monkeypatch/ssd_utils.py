from transformers import (
    PretrainedConfig,
    TrainerCallback,
)
import logging
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process
import axolotl
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import types
import math
import wandb
import transformers

logger = LOG = logging.getLogger("axolotl.monkeypatch.ssd")


# class NoisyTopkRouter(nn.Module):
#     def __init__(self, n_embed, num_experts, top_k_group):
#         super(NoisyTopkRouter, self).__init__()
#         self.top_k_group = top_k_group
#         self.topkroute_linear = nn.Linear(n_embed, num_experts)
#         nn.init.zeros_(self.topkroute_linear.weight)
#         # add noise
#         self.noise_linear =nn.Linear(n_embed, num_experts)
#         nn.init.zeros_(self.noise_linear.weight)
#         self.num_experts = num_experts

    
#     def forward(self, mh_output):
#         mh_output = mh_output.view(-1, mh_output.size(-1))

#         # mh_ouput is the output tensor from multihead self attention block
#         logits = self.topkroute_linear(mh_output)

#         #Noise logits
#         noise_logits = self.noise_linear(mh_output)

#         #Adding scaled unit gaussian noise to the logits
#         noise = torch.randn_like(logits)*F.softplus(noise_logits)
#         noisy_logits = logits + noise

#         routing_weights = F.softmax(noisy_logits, dim=-1)

#         routing_weights, group_mask = routing_weights.topk(self.top_k_group, dim=-1)

#         routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

#         routing_weights = routing_weights.to(mh_output.dtype)

#         group_mask = torch.nn.functional.one_hot(group_mask, num_classes=self.num_experts).permute(2, 1, 0)

#         return routing_weights, group_mask

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

class SelectKGroups(nn.Module):
    def __init__(self, num_group, top_k_group, hidden_size, resnet_num):
        super(SelectKGroups, self).__init__()
        # 初始化可训练的选择权重矩阵，大小为 (k_num_group, num_group)
        # self.selection_weights = nn.Parameter(torch.rand(top_k_group, num_group))
    
        self.resnet_block = nn.Sequential(
            *[ResBlock(hidden_size) for _ in range(resnet_num)]
        )

        # self.router_up_proj = nn.Linear(num_group, 128, bias=False)
        # self.router_act = nn.SiLU()
        # self.router_down_proj = nn.Linear(128, top_k_group, bias=False)

        self.selection_layer = nn.Linear(num_group, top_k_group, bias=False)

    
    def forward(self, all_hidden_states):
        
        all_hidden_states = self.resnet_block(all_hidden_states).to(all_hidden_states.dtype)

        # 通过矩阵乘法选择k_num_group
        # 首先将 all_hidden_states 转置为 (1, bs*seqlen, hidden_dim, num_group)
        all_hidden_states = all_hidden_states.permute(1, 2, 3, 0).contiguous()
        
        # 使用 selection_weights 从 num_group 中选择 k_num_group
        # selected_output = torch.matmul(all_hidden_states, self.selection_weights.transpose(0, 1)).to(all_hidden_states.dtype)
        selected_output = self.selection_layer(all_hidden_states).to(all_hidden_states.dtype)
        # selected_output = self.router_down_proj(self.router_act(self.router_up_proj(all_hidden_states))).to(all_hidden_states.dtype)
        
        # 输出形状为 (bs*seqlen, hidden_dim, k_num_group)
        return selected_output

def add_router(self):

    self.router = SelectKGroups(self.num_hidden_layers - self.top_layers_len, self.top_k_group, self.hidden_size, self.resnet_num)

    self.router.to(self.dtype).to(self.device)



def replace_compute_loss_kl_div(
    distill_temperature=2.0,
    distill_alpha=0.5,
    ssd_logging=False,
):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """

        log = {}

        outputs = model(
            **inputs,
        )

        draft_logits = outputs[1]
        draft_loss = outputs[0]
        base_logits = outputs[2]
        
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(draft_logits / distill_temperature, dim=-1),
                F.softmax(base_logits / distill_temperature, dim=-1),
            )
            * (distill_temperature ** 2)
        )

        loss = distill_alpha * draft_loss + (1.0 - distill_alpha) * loss_logits

        log[f"draft_loss"] = loss.item()
        
        # self.log(log)
        # Add prefix to the log
        if model.training:
            prefix = "train"
        else:
            prefix = "eval"
        log = {f"{prefix}/{k}": v for k, v in log.items()}
        if ssd_logging and self.state.is_world_process_zero:
            # Hardcoded for now
            wandb.log({
                **log,
                "train/global_step": self.state.global_step,
            })
        return (loss, draft_logits) if return_outputs else loss
    transformers.trainer.Trainer.compute_loss = compute_loss

def replace_compute_loss_kl_div_group(
    ssd_groups_coefficient,
    ssd_decay_coefficient, 
    distill_temperature=2.0,
    ssd_scheduler="constant",
    ssd_logging=False,
):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """

        log = {}

        outputs = model(
            **inputs,
        )
        loss = 0
        base_logits = outputs[1][0]
        all_logits = outputs[1][1:]
        num_group = all_logits.shape[0]
        for i in range(num_group):

            ssd_logits = all_logits[i]
            
            loss_function = nn.KLDivLoss(reduction="batchmean")
            loss_logits = (
                loss_function(
                    F.log_softmax(ssd_logits / distill_temperature, dim=-1),
                    F.softmax(base_logits / distill_temperature, dim=-1),
                )
                * (distill_temperature ** 2)
            )

            loss_i = loss_logits

            # Compute the coefficient for ssd losses
            if ssd_scheduler == "sine":
                ssd_scheduler_coefficient = math.sin(
                    self.state.global_step / self.state.max_steps * math.pi / 2
                )
            elif ssd_scheduler == "linear":
                ssd_scheduler_coefficient = (
                    self.state.global_step / self.state.max_steps
                )
            elif ssd_scheduler == "constant":
                ssd_scheduler_coefficient = 1
            elif ssd_scheduler.startswith("sine"):
                ratio = float(ssd_scheduler.split("_")[1])
                if self.state.global_step / self.state.max_steps < ratio:
                    ssd_scheduler_coefficient = math.sin(
                        self.state.global_step / self.state.max_steps / ratio * math.pi / 2
                    )
                else:
                    ssd_scheduler_coefficient = 1
            else:
                raise ValueError(
                    f"Invalid ssd_scheduler: {ssd_scheduler}. "
                    "Must be one of 'sine', 'linear', or 'constant'."
                )
            # Add decay coefficient to the loss
            if i == 0:
                loss += loss_i
            else:
                loss += loss_i * ssd_decay_coefficient ** i * ssd_groups_coefficient * ssd_scheduler_coefficient
        
        # self.log(log)
        # Add prefix to the log
        if model.training:
            prefix = "train"
        else:
            prefix = "eval"
        log = {f"{prefix}/{k}": v for k, v in log.items()}
        if ssd_logging and self.state.is_world_process_zero:
            # Hardcoded for now
            wandb.log({
                **log,
                "train/global_step": self.state.global_step,
            })
        return (loss, all_logits) if return_outputs else loss
    transformers.trainer.Trainer.compute_loss = compute_loss

def replace_compute_loss_cross_entropy(
    ssd_groups_coefficient,
    ssd_decay_coefficient, 
    ssd_scheduler="constant",
    ssd_logging=False,
):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """

        outputs = model(
            **inputs,
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        all_logits = outputs[1]
        num_group = all_logits.shape[0]
        for i in range(num_group):
            ssd_logits = all_logits[i, :, : -(1 + i)].contiguous()
            ssd_labels = labels[..., 1 + i :].contiguous()
            ssd_logits = ssd_logits.view(-1, all_logits.shape[-1])
            ssd_labels = ssd_labels.view(-1)
            ssd_labels = ssd_labels.to(ssd_logits.device)
            
            loss_i = loss_fct(ssd_logits, ssd_labels)
            # Compute the coefficient for ssd losses
            if ssd_scheduler == "sine":
                ssd_scheduler_coefficient = math.sin(
                    self.state.global_step / self.state.max_steps * math.pi / 2
                )
            elif ssd_scheduler == "linear":
                ssd_scheduler_coefficient = (
                    self.state.global_step / self.state.max_steps
                )
            elif ssd_scheduler == "constant":
                ssd_scheduler_coefficient = 1
            elif ssd_scheduler.startswith("sine"):
                ratio = float(ssd_scheduler.split("_")[1])
                if self.state.global_step / self.state.max_steps < ratio:
                    ssd_scheduler_coefficient = math.sin(
                        self.state.global_step / self.state.max_steps / ratio * math.pi / 2
                    )
                else:
                    ssd_scheduler_coefficient = 1
            else:
                raise ValueError(
                    f"Invalid ssd_scheduler: {ssd_scheduler}. "
                    "Must be one of 'sine', 'linear', or 'constant'."
                )
            # Add decay coefficient to the loss
            if i == 0:
                loss += loss_i
            else:
                loss += loss_i * ssd_decay_coefficient ** i * ssd_groups_coefficient * ssd_scheduler_coefficient


            not_ignore = ssd_labels.ne(IGNORE_TOKEN_ID)
            ssd_labels = ssd_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 10):
                _, topk = ssd_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(ssd_labels.unsqueeze(-1)).any(-1)
                log[f"draft_group{i}_top{k}"] = correct.float().mean().item()

            log[f"draft_group{i}_loss"] = loss_i.item()
            log["ssd_scheduler_coefficient"] = ssd_scheduler_coefficient
        # self.log(log)
        # Add prefix to the log
        if model.training:
            prefix = "train"
        else:
            prefix = "eval"
        log = {f"{prefix}/{k}": v for k, v in log.items()}
        if ssd_logging and self.state.is_world_process_zero:
            # Hardcoded for now
            wandb.log({
                **log,
                "train/global_step": self.state.global_step,
            })
        return (loss, all_logits) if return_outputs else loss
    transformers.trainer.Trainer.compute_loss = compute_loss

def replace_create_optimizer(
    router_lr_multiplier,
):
    # Copy from transformers.Trainer.create_optimizer
    from transformers.trainer import is_sagemaker_mp_enabled, Trainer, ShardedDDPOption
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            # Separately set lr for medusa_head
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "router" not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "router" in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * router_lr_multiplier,
                },
                
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    transformers.trainer.Trainer.create_optimizer = create_optimizer

    # Fix deepspeed's optimizer
    def deepspeed_init(trainer, num_training_steps, inference=False):
        """
        Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

        If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

        Args:
            trainer: Trainer object
            num_training_steps: per single gpu
            resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
            inference: launch in inference mode (no optimizer and no lr scheduler)

        Returns: optimizer, lr_scheduler

        We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
        https://github.com/microsoft/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
        can't resume from a checkpoint after it did some stepping https://github.com/microsoft/DeepSpeed/issues/1612

        """
        from deepspeed.utils import logger as ds_logger
        from transformers.integrations.deepspeed import deepspeed_optim_sched

        model = trainer.model
        args = trainer.args

        hf_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config

        # resume config update - some bits like `model` and `num_training_steps` only become available during train
        hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

        # set the Deepspeed log level consistent with the Trainer
        ds_logger.setLevel(args.get_process_log_level())

        if inference:
            # only Z3 makes sense for the inference
            if not hf_deepspeed_config.is_zero3():
                raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")

            # in case the training config is re-used for inference
            hf_deepspeed_config.del_config_sub_tree("optimizer")
            hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
            optimizer, lr_scheduler = None, None
            model_parameters = None
        else:
            trainer.optimizer = None  # important for when deepspeed_init is used as re-init
            self = trainer
            opt_model = model
            decay_parameters = self.get_decay_parameter_names(opt_model)
            model_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "router" not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "router" in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * router_lr_multiplier,
                },
                
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            
            # list(filter(lambda p: p.requires_grad, model.parameters()))
            optimizer, lr_scheduler = deepspeed_optim_sched(
                trainer, hf_deepspeed_config, args, num_training_steps, model_parameters
            )

        # keep for quick debug:
        # from pprint import pprint; pprint(config)

        return optimizer, lr_scheduler
    transformers.integrations.deepspeed.deepspeed_init = deepspeed_init