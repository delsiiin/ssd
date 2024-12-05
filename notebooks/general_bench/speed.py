import json
from transformers import AutoTokenizer
import numpy as np

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, help="Model name or path.", default='/root/MODELS/vicuna-7b-v1.3')
    parser.add_argument("--model_id", type=str, required=False)
    parser.add_argument(
        "--bench_name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )

    # ssd
    parser.add_argument("--davm", action='store_true', required=False, default=False)

    parser.add_argument("--early_exit", action='store_true', required=False, default=False)

    parser.add_argument("--attn", action='store_true', required=False, default=False)

    parser.add_argument("--medusa", action='store_true', required=False, default=False)

    args = parser.parse_args()

    tokenizer=AutoTokenizer.from_pretrained(args.model_name)
    if args.davm:
        jsonl_file = f"./result/{args.bench_name}/{args.model_id}_davm.jsonl"
    elif args.early_exit:
        jsonl_file = f"./result/{args.bench_name}/{args.model_id}_ee.jsonl"
    elif args.attn:
        jsonl_file = f"./result/{args.bench_name}/{args.model_id}_attn.jsonl"
    elif args.medusa:
        jsonl_file = f"./result/{args.bench_name}/{args.model_id}_medusa.jsonl"
    
    jsonl_file_base = f"./result/{args.bench_name}/{args.model_id}.jsonl"
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)



    speeds=[]
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds.append(tokens/times)


    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)


    total_time=0
    total_token=0
    speeds0=[]
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens = 0
        for i in answer:
            tokens += (len(tokenizer(i).input_ids) - 1)
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds0.append(tokens / times)
        total_time+=times
        total_token+=tokens



    # print('speed',np.array(speeds).mean())
    # print('speed0',np.array(speeds0).mean())
    print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())