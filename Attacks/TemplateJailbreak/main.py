import pandas as pd
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
from utils import model_names_list, get_model_path
from global_config import get_config  

config = get_config()
REPEAT_TIME_PER_QUESTION = config.REPEAT_TIME_PER_QUESTION
    
def run(model_name):
    f = open(f'../../Data/data_main.json',) 
    datas = json.load(f) 
    f.close() 
    results = [{} for _ in range(len(datas))]

    if args.model in model_names_list.keys():
        model_name = model_names_list[args.model]
        model_path = get_model_path(model_name)

        print(f"\n\n\nmodelPath: {model_path}\n\n\n")
        model_name_absolute = "/".join(model_path.split("/")[-2:])
    else:
        model_name = 'unknown'
        raise ValueError(f"Unknown model name, Available models are {model_names_list.keys()}")

    local_model = models.LocalVLLM(model_path=model_name_absolute, model_name=model_name)

        
    final_results = []
    prompts = []
    for idx, data in enumerate(datas):
        question_bkp = data['questions']
        questions = data['questions']
        CURRENT_ITERATION = 0
        while CURRENT_ITERATION < REPEAT_TIME_PER_QUESTION:
            results[idx]['topic'] = data['topic']
            results[idx]['qA_pairs'] = []
            for i, question in enumerate(questions):
                print(f"Question {(idx * len(questions) * REPEAT_TIME_PER_QUESTION + i + 1) + CURRENT_ITERATION * len(questions)} / {len(questions) * REPEAT_TIME_PER_QUESTION * len(datas)}")
                
                prompt = data['inception_attack'].replace("ඞ", question)
                prompts.append(local_model.create_conv_prompt(prompt))
                # print(f"prompt: {prompt}")
                # print(f"prompts: {prompts}")
                target_response_list = local_model.generate_batch(prompts, max_tokens = 2048)
                # target_response_list = targetLM.get_response(data['inception_attack'].replace("ඞ", question), args.defense)
                results[idx]['qA_pairs'].append({'Q': question, 'A': target_response_list})
                # print(f"results[idx]['qA_pairs'][-1]: {results[idx]['qA_pairs'][-1]}")
                # if i == 0:
                #     continue
                # final_results.append({'prompt': data['inception_attack'], 'response': target_response_list[0], 'question': question_bkp[i-1],"iteration":CURRENT_ITERATION })
                final_results.append({'prompt': data['inception_attack'], 'response': target_response_list[0], 'question': question_bkp[i],"iteration":CURRENT_ITERATION })
                # print(f"final_results[-1]: {final_results[-1]}")
            CURRENT_ITERATION+= 1
        
    
    model_name_path = model_name.replace("/","_")
    if not os.path.exists(f"/content/drive/MyDrive/llm_attack_arena/Attacks/DeepInception/Results"):
            os.makedirs(f"/content/drive/MyDrive/llm_attack_arena/Attacks/DeepInception/Results")
    with open(f'/content/drive/MyDrive/llm_attack_arena/Attacks/DeepInception/Results/DeepInception_{model_name_path}.json', 'w') as f:
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
    args = parser.parse_args()
    run(args.model)