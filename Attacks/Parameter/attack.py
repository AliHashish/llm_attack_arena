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
from utils import model_names_list, get_model_path, get_config

config = get_config()
REPEAT_TIME_PER_QUESTION = config.REPEAT_TIME_PER_QUESTION
# Repeat each question 3 times (to account for model randomness)

    
def run(model_name):
    f = open(f'../../Data/data.csv')
    datas = pd.read_csv(f, usecols=['goal'])
    f.close()

    if args.model in model_names_list.keys():
        model_name = model_names_list[args.model]
        model_path = get_model_path(model_name)

        print(f"\n\n\nmodelPath: {model_path}\n\n\n")    # print to make sure it's correct
        model_name_absolute = "/".join(model_path.split("/")[-2:])
    else:
        model_name = 'unknown'
        raise ValueError(f"Unknown model name, Available models are {model_names_list.keys()}")
    
    # Load the model
    local_model = models.LocalVLLM(model_path=model_name_absolute, model_name=model_name, gpu_memory_utilization=0.6)
        
    final_results = []
    prompts = []

    # The parameter ranges
    temps = np.arange(0.05, 1.05, 0.1)
    temps = [round(float(t), 2) for t in temps]

    top_ps = np.arange(0.05, 1.05, 0.1)
    top_ps = [round(float(t), 2) for t in top_ps]

    top_ks = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    top_presences = np.arange(-2, 2.1, 0.5)
    top_presences = [float(t) for t in top_presences]

    top_frequencies = np.arange(-2, 2.1, 0.5)
    top_frequencies = [float(t) for t in top_frequencies]

    # For each question, we will loop over all parameters 3 times
    total_questions = len(datas.values) * REPEAT_TIME_PER_QUESTION * (len(temps) + len(top_ps) + len(top_ks) + len(top_presences) + len(top_frequencies))
    current_question = 0

    #Temperature
    for temperature in temps:
        # CURRENT_ITERATION = 0
        for idx, question_list in enumerate(datas.values):
            prompts = []
            prompt = question_list[0]
            
            # CURRENT_ITERATION += 1
            current_question += REPEAT_TIME_PER_QUESTION

            print(f"Question {current_question}/{total_questions}")

            # Create the conversation prompt
            conv_prompt = local_model.create_conv_prompt(prompt)

            prompts.append(conv_prompt)
            # Generate responses in batch
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, temperature=temperature, n=REPEAT_TIME_PER_QUESTION)

            for i in range(REPEAT_TIME_PER_QUESTION):
                final_results.append({'prompt': prompt, 'response': target_response_list[i], 'question': prompt,"param":{'temperature':temperature},"iteration":i+1 })


    #Top-p
    for top_p in top_ps:
        for idx, question_list in enumerate(datas.values):
            prompts = []
            prompt = question_list[0]

            current_question += REPEAT_TIME_PER_QUESTION

            print(f"Question {current_question}/{total_questions}")

            # Conversation prompt
            prompts.append(local_model.create_conv_prompt(prompt))

            # Generate responses
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, top_p=top_p, n=REPEAT_TIME_PER_QUESTION)

            for i in range(REPEAT_TIME_PER_QUESTION):
                final_results.append({'prompt': prompt, 'response': target_response_list[i], 'question': prompt,"param":{'top_p':top_p},"iteration":i+1 })


    #Top-k
    for top_k in top_ks:
        for idx, question_list in enumerate(datas.values):
            prompts = []
            prompt = question_list[0]

            current_question += REPEAT_TIME_PER_QUESTION

            print(f"Question {current_question}/{total_questions}")

            # Conversation prompt
            prompts.append(local_model.create_conv_prompt(prompt))
            
            # Generate responses
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, top_k=top_k, n=REPEAT_TIME_PER_QUESTION)
            
            for i in range(REPEAT_TIME_PER_QUESTION):
                final_results.append({'prompt': prompt, 'response': target_response_list[i], 'question': prompt,"param":{'top_k':top_k},"iteration":i+1 })



    #Presence
    for presence in top_presences:
        for idx, question_list in enumerate(datas.values):
            prompts = []
            prompt = question_list[0]

            current_question += REPEAT_TIME_PER_QUESTION

            print(f"Question {current_question}/{total_questions}")


            # Conversation prompt
            prompts.append(local_model.create_conv_prompt(prompt))
            
            # Generate responses
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, presence_penalty=presence, n=REPEAT_TIME_PER_QUESTION)
            
            for i in range(REPEAT_TIME_PER_QUESTION):
                final_results.append({'prompt': prompt, 'response': target_response_list[i], 'question': prompt,"param":{'presence':presence},"iteration":i+1 })

    #Frequency
    for frequency in top_frequencies:
        for idx, question_list in enumerate(datas.values):
            prompts = []
            prompt = question_list[0]

            current_question += REPEAT_TIME_PER_QUESTION

            print(f"Question {current_question}/{total_questions}")

            # Conversation prompt
            conv_prompt = local_model.create_conv_prompt(prompt)
            
            prompts.append(conv_prompt)
            
            # Generate responses
            target_response_list = local_model.generate_batch(prompts, max_tokens = 1500, frequency_penalty=frequency, n=REPEAT_TIME_PER_QUESTION)
            
            for i in range(REPEAT_TIME_PER_QUESTION):
                final_results.append({'prompt': prompt, 'response': target_response_list[i], 'question': prompt,"param":{'frequency':frequency},"iteration":i+1 })



    # Saving the results to a JSON file
    model_name_path = model_name.replace("/","_")
    if not os.path.exists(f"/content/drive/MyDrive/llm_attack_arena/Attacks/Parameter/Results"):
            os.makedirs(f"/content/drive/MyDrive/llm_attack_arena/Attacks/Parameter/Results")
    with open(f'/content/drive/MyDrive/llm_attack_arena/Attacks/Parameter/Results/Parameters_{model_name_path}.json', 'w') as f:
         json.dump(final_results, f, indent=4)
    
    # Freeing memory
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
    run(args.model)