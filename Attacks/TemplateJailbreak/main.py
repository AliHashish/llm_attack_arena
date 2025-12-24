import pandas as pd

import json

import re
import json
import argparse
import sys
import os
parent_parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_parent_dir)
import models
import torch
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import model_names_list, get_model_path, get_config

config = get_config()
# MAX_ALLOWED_ITERATION_PER_QUESTION = config.MAX_ALLOWED_ITERATION_PER_QUESTION
REPEAT_TIME_PER_QUESTION = config.REPEAT_TIME_PER_QUESTION

final_res = []
    
def run(model_name):
    file_path = '../../Data/data.csv'
    jailbreak_path = '../../Data/jailbreak-prompt.xlsx'

    if args.model in model_names_list.keys():
        model_name = model_names_list[args.model]
        model_path = get_model_path(model_name)

        print(f"\n\n\nmodelPath: {model_path}\n\n\n")
        model_name_absolute = "/".join(model_path.split("/")[-2:])
    else:
        model_name = 'unknown'
        raise ValueError(f"Unknown model name, Available models are {model_names_list.keys()}")
    
    local_model = models.LocalVLLM(model_path=model_name_absolute, model_name=model_name)

    df1 = pd.read_excel(jailbreak_path)
    prompts_templates = df1['text']
    train_data = pd.read_csv(file_path)
    questions = train_data['goal'].tolist()
        
        
    for question in questions:
        print(f"running {question}")
        prompts = []
        templates= []
        original_prompts = []

        for template in prompts_templates:
            prompt = template.replace('[INSERT PROMPT HERE]', question)
            original_prompts.extend([prompt]*REPEAT_TIME_PER_QUESTION)
            prompts.append(local_model.create_conv_prompt(prompt))
            templates.extend([template]*REPEAT_TIME_PER_QUESTION)

        responses = local_model.generate_batch(prompts,n=REPEAT_TIME_PER_QUESTION,temperature=1)

        i = 0
        for prompt,template,response in zip(original_prompts,templates,responses):
            i = i%REPEAT_TIME_PER_QUESTION + 1
            if i == 0:
                i = REPEAT_TIME_PER_QUESTION
            final_res.append({"template":template,"question":question,"prompt":prompt,"response":response,"iteration":i})
            

    model_name_path = model_name.replace("/", "_")
    if not os.path.exists(f"/content/drive/MyDrive/llm_attack_arena/Attacks/TemplateJailbreak/Results"):
        os.makedirs(f"/content/drive/MyDrive/llm_attack_arena/Attacks/TemplateJailbreak/Results")
    with open(f'/content/drive/MyDrive/llm_attack_arena/Attacks/TemplateJailbreak/Results/TemplateJailbreak_{model_name_path}.json', 'w') as f:
        json.dump(final_res, f, indent=4)

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
    args = parser.parse_args()
    run(args.model)