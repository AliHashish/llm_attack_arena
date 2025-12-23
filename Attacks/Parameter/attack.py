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
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import model_names_list, get_model_path
from global_config import get_config  

config = get_config()
REPEAT_TIME_PER_QUESTION = config.REPEAT_TIME_PER_QUESTION

    
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
    

    local_model = models.LocalVLLM(model_path=model_name_absolute, model_name=model_name, gpu_memory_utilization=0.6)
        
    final_results = []
    prompts = []

    temps = np.arange(0.05, 1.05, 0.1)
    temps = [round(float(t), 2) for t in temps]

    top_ps = np.arange(0.05, 1.05, 0.1)
    top_ps = [round(float(t), 2) for t in top_ps]

    top_ks = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    top_presences = np.arange(-2, 2.1, 0.5)
    top_presences = [float(t) for t in top_presences]

    top_frequencies = np.arange(-2, 2.1, 0.5)
    top_frequencies = [float(t) for t in top_frequencies]

    total_questions = len(datas.values) * REPEAT_TIME_PER_QUESTION * (len(temps) + len(top_ps) + len(top_ks) + len(top_presences) + len(top_frequencies))
    current_question = 0

    #Temperature
    print("Starting Temperature Attack")
    for temperature in temps:
        # CURRENT_ITERATION = 0
        print(f"Trying Temperature: {temperature}")
        for idx, question_list in enumerate(datas.values):
            prompts = []
            print(f"Question list {question_list}")
            prompt = question_list[0]
            
            # CURRENT_ITERATION += 1
            current_question += REPEAT_TIME_PER_QUESTION

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
    for top_p in top_ps:
        print(f"Trying Top-p: {top_p}")
        for idx, question_list in enumerate(datas.values):
            prompts = []
            prompt = question_list[0]

            current_question += REPEAT_TIME_PER_QUESTION

            print(f"Question {current_question}/{total_questions}")

            prompts.append(local_model.create_conv_prompt(prompt))

            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, top_p=top_p, n=REPEAT_TIME_PER_QUESTION)

            for i in range(REPEAT_TIME_PER_QUESTION):
                final_results.append({'prompt': prompt, 'response': target_response_list[i], 'question': prompt,"param":{'top_p':top_p},"iteration":i+1 })


    #Top-k
    print("Starting Top-k Attack")
    for top_k in top_ks:
        print(f"Trying Top-k: {top_k}")
        for idx, question_list in enumerate(datas.values):
            prompts = []
            prompt = question_list[0]

            current_question += REPEAT_TIME_PER_QUESTION

            print(f"Question {current_question}/{total_questions}")


            prompts.append(local_model.create_conv_prompt(prompt))
            
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, top_k=top_k, n=REPEAT_TIME_PER_QUESTION)
            
            for i in range(REPEAT_TIME_PER_QUESTION):
                final_results.append({'prompt': prompt, 'response': target_response_list[i], 'question': prompt,"param":{'top_k':top_k},"iteration":i+1 })



    #Presence
    print("Starting Presence Attack")
    for presence in top_presences:
        print(f"Trying Presence: {presence}")
        for idx, question_list in enumerate(datas.values):
            prompts = []
            prompt = question_list[0]

            current_question += REPEAT_TIME_PER_QUESTION

            print(f"Question {current_question}/{total_questions}")


            prompts.append(local_model.create_conv_prompt(prompt))
            
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, presence_penalty=presence, n=REPEAT_TIME_PER_QUESTION)
            
            for i in range(REPEAT_TIME_PER_QUESTION):
                final_results.append({'prompt': prompt, 'response': target_response_list[i], 'question': prompt,"param":{'presence':presence},"iteration":i+1 })

    #Frequency
    print("Starting Frequency Attack")
    for frequency in top_frequencies:
        print(f"Trying Frequency: {frequency}")
        for idx, question_list in enumerate(datas.values):
            prompts = []
            prompt = question_list[0]

            current_question += REPEAT_TIME_PER_QUESTION

            print(f"Question {current_question}/{total_questions}")

            conv_prompt = local_model.create_conv_prompt(prompt)
            
            prompts.append(conv_prompt)
            
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, frequency_penalty=frequency, n=REPEAT_TIME_PER_QUESTION)
            
            for i in range(REPEAT_TIME_PER_QUESTION):
                final_results.append({'prompt': prompt, 'response': target_response_list[i], 'question': prompt,"param":{'frequency':frequency},"iteration":i+1 })




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
        help="model name to be used for the attack"
    )
    args = parser.parse_args()
    process_raw_jailbreak_prompts(args.model)