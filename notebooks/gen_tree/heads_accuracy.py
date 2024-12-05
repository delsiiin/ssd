import os
import torch
import json
from contextlib import contextmanager
import numpy as np
# from medusa.model.medusa_model import MedusaModel
from medusa.model.configuration_llama_ssd import LlamaConfig
from medusa.model.kv_cache import *
# from medusa.model.utils import *
from medusa.model.medusa_choices import *
from copy import deepcopy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastchat.model.model_adapter import get_conversation_template
from tqdm import tqdm
import argparse

from peft import PeftModel, PeftConfig

from utils import *

from transformers import AutoTokenizer

def get_accuracies(ssd, logit):
    # get the correct counts of each head
    seq_len, choices, topk = ssd.shape
    results = []
    for choice in range(choices):
        results.append(ssd[:-choice - 1,choice].eq(logit[choice + 1:,0]))
    return results



def main(args):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用，使用 CPU")

    config_kwargs = {}

    config_kwargs["top_layers_len"] = args.top_layers_len 
    config_kwargs["top_k_group"] = args.top_k_group 
    config_kwargs["resnet_num"] = args.resnet_num 

    config = LlamaConfig.from_pretrained(args.model_path, **config_kwargs)
    
    # print(config.num_skipped_draft_model)

    lora_path = f"/home/zmw/ssd/axolotl/models/vicuna-7b-v1.3-qlora-ssd-out-router-top-{args.top_layers_len}-k-{args.top_k_group}-ee-only-seq-2048-stage-1-smts"
    lora_config = PeftConfig.from_pretrained(lora_path)


    if args.early_exit:
        from medusa.model.modeling_llama_ssd_v1_top_layers_router_last_tree import LlamaForCausalLM, add_router
    elif args.davm:
        from medusa.model.modeling_llama_ssd_router_dmlp_vattn import LlamaForCausalLM, add_router
    elif args.attn:
        from medusa.model.modeling_llama_ssd_v1_top_layers_router_last_attn import LlamaForCausalLM, add_router
    else:
        from medusa.model.modeling_llama_ssd_router import LlamaForCausalLM, add_router

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        config=config,
    )

    model = model.eval().to(device)

    if args.attn:
        add_router(model, args.attn_hid_dim)
    else:
        add_router(model)

    model = PeftModel.from_pretrained(model, lora_path)

    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)


    data = json.load(open(args.data_path))
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    if args.attn:
        past_key_values_attn, past_key_values_data_attn, current_length_data_attn = initialize_past_key_values_attn(model, args.top_k_group, args.attn_hid_dim)

        model.past_key_values_attn = past_key_values_attn
        model.past_key_values_data_attn = past_key_values_data_attn
        model.current_length_data_attn = current_length_data_attn

    results = None

    for sample in tqdm((data)):
        conv = get_conversation_template("vicuna")
        conv.messages = []
        conv.append_message(conv.roles[0], sample["instruction"])
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        steps = args.steps
        logits_ids = []
        ssd_topk_ids = []

        with torch.inference_mode():
            input_ids = tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()
            model.current_length_data.zero_() # this is for rerun
            if args.attn:
                model.current_length_data_attn.zero_() # this is for rerun

            reset_ssd_mode(model)
           
            with model.self_draft():
            # print(draft_attn_skip_masks)
                if args.attn:
                    draft_logits, logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values, past_key_values_attn=model.past_key_values_attn, top_layers_len=args.top_layers_len)
                else:
                    draft_logits, logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values, top_layers_len=args.top_layers_len)
        
            _, ssd_topk = draft_logits[...,-1,:].topk(20, dim=-1)

            input_id = logits[:, -1:].argmax(dim=-1)
            logits_ids.append(input_id.detach().cpu())
            ssd_topk_ids.append(ssd_topk.detach().cpu())
            for _ in range(steps):
                with model.self_draft():
                    # print(draft_attn_skip_masks)
                    if args.attn:
                        draft_logits, logits = model(input_id, output_orig = True, past_key_values=model.past_key_values, past_key_values_attn=model.past_key_values_attn, top_layers_len=args.top_layers_len)
                    else:
                        draft_logits, logits = model(input_id, output_orig = True, past_key_values=model.past_key_values, top_layers_len=args.top_layers_len)
        
                _, ssd_topk = draft_logits[...,-1,:].topk(20, dim=-1)
                input_id = logits[:, -1:].argmax(dim=-1)
                logits_ids.append(input_id.detach().cpu())
                ssd_topk_ids.append(ssd_topk.detach().cpu())
            logits_ids = torch.stack(logits_ids, dim=0)
            ssd_topk_ids = torch.stack(ssd_topk_ids, dim=0).squeeze(2)
            if results is None:
                results = get_accuracies(ssd_topk_ids, logits_ids)
            else:
                # cat sub results
                cur_results = get_accuracies(ssd_topk_ids, logits_ids)
                for i in range(len(results)):
                    results[i] = torch.cat((results[i], cur_results[i]), dim=0)

    save_path = os.path.join(args.save_dir, args.model_name + "-" + str(args.top_layers_len) + "-" + str(args.top_k_group) + "_heads_accuracy.pt")
    torch.save(results, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSD Model Evaluator")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained Medusa model.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the evaluation data in JSON format.")
    parser.add_argument("--save_dir", type=str, default="./data",
                        help="Directory to save the results.")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of steps to run the model.")
    

    parser.add_argument("--top_layers_len", type=int, required=False, help="top layers to keep", default=20)
    parser.add_argument("--top_k_group", type=int, required=False, help="Draft group num.", default=4)
    parser.add_argument("--resnet_num", type=int, required=False, help="resnet num.", default=1)
    parser.add_argument("--early_exit", action='store_true', required=False, default=False)
    parser.add_argument("--davm", action='store_true', required=False, default=False)
    parser.add_argument("--attn", action='store_true', required=False, default=False)
    parser.add_argument("--attn_hid_dim", type=int, required=True, default=1024)

    args = parser.parse_args()

    # If the save directory doesn't exist, create it
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)