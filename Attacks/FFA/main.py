import pandas as pd

import json

import re
import json
from collections import defaultdict
import argparse
import sys
import os
parent_parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_parent_dir)
import models
import torch
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import model_names_list, get_model_path
from FFA_templates import FFA_templates


from global_config import get_config  
config = get_config()
# MAX_ALLOWED_ITERATION_PER_QUESTION = config.MAX_ALLOWED_ITERATION_PER_QUESTION
REPEAT_TIME_PER_QUESTION = config.REPEAT_TIME_PER_QUESTION

final_res = []
    
def process_raw_jailbreak_prompts(model_name,question_count):
    f = open(f'../../Data/data.csv')
    datas = pd.read_csv(f, usecols=['goal'])
    f.close()
    results = [{} for _ in range(len(datas))]

    if args.model in model_names_list.keys():
        model_name = model_names_list[args.model]
        model_path = get_model_path(model_name)
        directory_name = args.model

        print(f"\n\n\nmodelPath: {model_path}\n\n\n")
        model_name_absolute = "/".join(model_path.split("/")[-2:])
    else:
        model_name = 'unknown'
        raise ValueError(f"Unknown model name, Available models are {model_names_list.keys()}")
    

    local_model = models.LocalVLLM(model_path=model_name_absolute, model_name=model_name)
        
    final_results = []
    prompts = []
    for idx, question_list in enumerate(datas.values):
        question = question_list[0]
        CURRENT_ITERATION = 0
        for template in FFA_templates:
            CURRENT_ITERATION += 1
            results[idx]['qA_pairs'] = []

            print(f"Question {CURRENT_ITERATION + idx * len(FFA_templates)}/{len(datas) * len(FFA_templates)}")

            prompt = template.replace("ඞ", question)
            prompts.append(local_model.create_conv_prompt(prompt))
            # print(f"prompt: {prompt}")
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500)
            results[idx]['qA_pairs'].append({'Q': question, 'A': target_response_list})

            final_results.append({'prompt': template, 'response': target_response_list[0], 'question': question,"template number":CURRENT_ITERATION })
            # print(f"final_results[-1]: {final_results[-1]}")

        
    
    model_name_path = model_name.replace("/","_")
    if not os.path.exists(f"/content/drive/MyDrive/llm_attack_arena/Attacks/FFA/Results"):
            os.makedirs(f"/content/drive/MyDrive/llm_attack_arena/Attacks/FFA/Results")
    with open(f'/content/drive/MyDrive/llm_attack_arena/Attacks/FFA/Results/FFA_{model_name_path}.json', 'w') as f:
         json.dump(final_results, f, indent=4)
    
    del local_model
    gc.collect()
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="model name to be used for the attack",
    )
    parser.add_argument(
        "--question_count",
        type=int,
        default=100,
        help="how many questions you would like to test",
    )
    args = parser.parse_args()
    process_raw_jailbreak_prompts(args.model,args.question_count)