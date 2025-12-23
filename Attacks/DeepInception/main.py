import argparse
import json
import os

from conversers import load_attack_and_target_models

import sys
from global_config import get_config  
from utils import model_names_list


original_sys_path = sys.path.copy()
project_root_path = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(project_root_path)
config = get_config()
REPEAT_TIME_PER_QUESTION = config.REPEAT_TIME_PER_QUESTION
#a reset function to reset the sys.path
sys.path = original_sys_path

def run(model_name):
    

    final_results = []
    if args.target_model in model_names_list.keys():
        model_name = model_names_list[args.target_model]
    else:
        raise ValueError(f"Unknown model name, supports only {model_names_list.keys()}")
    
    f = open(f'./../../Data/data_main.json',) 
    datas = json.load(f) 
    f.close() 
    results = [{} for _ in range(len(datas))]
    targetLM = load_attack_and_target_models(args)
    
    for idx, data in enumerate(datas):
        question_bkp = data['questions']

        questions = data['questions']
        CURRENT_ITERATION = 0
        while CURRENT_ITERATION < REPEAT_TIME_PER_QUESTION:
            results[idx]['topic'] = data['topic']
            results[idx]['qA_pairs'] = []
            for i, question in enumerate(questions):
                print(f"Question {(idx * len(questions) * REPEAT_TIME_PER_QUESTION + i + 1) + CURRENT_ITERATION * len(questions)} / {len(questions) * REPEAT_TIME_PER_QUESTION * len(datas)}")
                target_response_list = targetLM.get_response(data['inception_attack'].replace("ඞ", question), args.defense)
                results[idx]['qA_pairs'].append({'Q': question, 'A': target_response_list})

                final_results.append({'prompt': data['inception_attack'], 'response': target_response_list[0], 'question': question_bkp[i],"iteration":CURRENT_ITERATION })
            CURRENT_ITERATION+= 1


    model_name_path = model_name.replace("/","_")
    if not os.path.exists(f"/content/drive/MyDrive/llm_attack_arena/Attacks/DeepInception/Results"):
            os.makedirs(f"/content/drive/MyDrive/llm_attack_arena/Attacks/DeepInception/Results")
    with open(f'/content/drive/MyDrive/llm_attack_arena/Attacks/DeepInception/Results/DeepInception_{model_name_path}.json', 'w') as f:
         json.dump(final_results, f, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ########### Target model parameters ##########
    parser.add_argument(
        "--model",
        choices= model_names_list.keys(),
        help = "model name to be used for the attack.",
    )
    args = parser.parse_args()

    run(args.model)