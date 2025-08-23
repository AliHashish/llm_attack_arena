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
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import model_names_list, get_model_path


from global_config import get_config  
config = get_config()
# MAX_ALLOWED_ITERATION_PER_QUESTION = config.MAX_ALLOWED_ITERATION_PER_QUESTION
REPEAT_TIME_PER_QUESTION = config.REPEAT_TIME_PER_QUESTION

final_res = []
    
def process_raw_jailbreak_prompts(model_name):
    f = open(f'../../Data/data.csv')
    datas = pd.read_csv(f, usecols=['goal'])
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

    temps = np.arange(0.05, 1.05, 0.1)
    top_ps = np.arange(0, 1.05, 0.1)
    top_ks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    top_presences = np.arange(-2, 2.1, 0.5)
    top_frequencies = np.arange(-2, 2.1, 0.5)
    total_questions = len(datas.values) * REPEAT_TIME_PER_QUESTION * \
    (args.tune_temp + args.tune_topp + args.tune_topk + args.tune_presence + args.tune_frequency)
    current_question = 0

    #Temperature
    print("Starting Temperature Attack")
    for temperature in temps:
        # CURRENT_ITERATION = 0
        print(f"Trying Temperature: {temperature}")
        for idx, question_list in enumerate(datas.values):
            print(f"Question list {question_list}")
            prompt = question_list[0]
            
            # CURRENT_ITERATION += 1
            current_question += REPEAT_TIME_PER_QUESTION
            results[idx]['qA_pairs'] = []

            print(f"Question {current_question}/{total_questions}")

            print(f"prompt: {prompt}")
            conv_prompt = local_model.create_conv_prompt(prompt)
            print(f"create_conv_prompt: {conv_prompt}")

            prompts.append(conv_prompt)
            print(f"\n\nana el prompts: {prompts}")
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, temperature=temperature, n=REPEAT_TIME_PER_QUESTION)
            print(f"length Response List: {len(target_response_list)}")
            # results[idx]['qA_pairs'].append({'Q': prompt, 'A': target_response_list})
            print(f"Response List: {target_response_list}")

            for i in range(REPEAT_TIME_PER_QUESTION):
                final_results.append({'prompt': prompt, 'response': target_response_list[i], 'question': prompt,"param":{'temperature':temperature},"iteration":i+1 })
            print(f"Final Results: {final_results[-1]}")

    #Top-p
    print("Starting Top-p Attack")
    for top_p in np.arange(0, 1.05, 0.1):
        CURRENT_ITERATION = 0
        print(f"Trying Top-p: {top_p}")
        for idx, question_list in enumerate(datas.values):
            prompt = question_list[0]

            CURRENT_ITERATION += 1
            results[idx]['qA_pairs'] = []

            print(f"Question {CURRENT_ITERATION}/60")

            prompts.append(local_model.create_conv_prompt(prompt))
            # print(f"prompt: {prompt}")
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, top_p=top_p)
            results[idx]['qA_pairs'].append({'Q': prompt, 'A': target_response_list})

            final_results.append({'prompt': prompt, 'response': target_response_list[0], 'question': prompt,"template number":CURRENT_ITERATION })


    #Top-k
    print("Starting Top-k Attack")
    for top_k in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        CURRENT_ITERATION = 0
        print(f"Trying Top-k: {top_k}")
        for idx, question_list in enumerate(datas.values):
            prompt = question_list[0]

            CURRENT_ITERATION += 1
            results[idx]['qA_pairs'] = []

            print(f"Question {CURRENT_ITERATION}/60")

            prompts.append(local_model.create_conv_prompt(prompt))
            # print(f"prompt: {prompt}")
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, top_k=top_k)
            results[idx]['qA_pairs'].append({'Q': prompt, 'A': target_response_list})

            final_results.append({'prompt': prompt, 'response': target_response_list[0], 'question': prompt,"template number":CURRENT_ITERATION })

    #Presence
    print("Starting Presence Attack")
    for presence in np.arange(-2, 2.1, 0.5):
        CURRENT_ITERATION = 0
        print(f"Trying Presence: {presence}")
        for idx, question_list in enumerate(datas.values):
            prompt = question_list[0]

            CURRENT_ITERATION += 1
            results[idx]['qA_pairs'] = []

            print(f"Question {CURRENT_ITERATION}/60")

            prompts.append(local_model.create_conv_prompt(prompt))
            # print(f"prompt: {prompt}")
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, presence_penalty=presence)
            results[idx]['qA_pairs'].append({'Q': prompt, 'A': target_response_list})

            final_results.append({'prompt': prompt, 'response': target_response_list[0], 'question': prompt,"template number":CURRENT_ITERATION })

    #Frequency
    for frequency in np.arange(-2, 2.1, 0.5):
        CURRENT_ITERATION = 0
        print(f"Trying Frequency: {frequency}")
        for idx, question_list in enumerate(datas.values):
            prompt = question_list[0]

            CURRENT_ITERATION += 1
            results[idx]['qA_pairs'] = []

            print(f"Question {CURRENT_ITERATION}/60")
            print(f"prompt: {prompt}")
            conv_prompt = local_model.create_conv_prompt(prompt)
            print(f"create_conv_prompt: {conv_prompt}")
            prompts.append(conv_prompt)
            # print(f"prompt: {prompt}")
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, frequency_penalty=frequency)
            results[idx]['qA_pairs'].append({'Q': prompt, 'A': target_response_list})
            print(f"Response List: {target_response_list}")

            final_results.append({'prompt': prompt, 'response': target_response_list[0], 'question': prompt,"template number":CURRENT_ITERATION })



    model_name_path = model_name.replace("/","_")
    if not os.path.exists(f"/content/drive/MyDrive/llm_attack_arena/Attacks/Parameter/Results"):
            os.makedirs(f"/content/drive/MyDrive/llm_attack_arena/Attacks/Parameter/Results")
    with open(f'/content/drive/MyDrive/llm_attack_arena/Attacks/Parameter/Results/Parameters_{model_name_path}.json', 'w') as f:
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
    process_raw_jailbreak_prompts(args.model)