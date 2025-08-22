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
# # Set the path to the 'papers' directory
# papers_path = '/workspace/papers'

# # Add the 'papers' directory to sys.path
# sys.path.append(papers_path)

# Now you can import the 'models' module
import models
import torch
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import model_names_list, get_model_path

# import sys

# original_sys_path = sys.path.copy()
# project_root_path = os.path.join(os.path.dirname(__file__), '../../')
# sys.path.append(project_root_path)
from global_config import get_config  
config = get_config()
# MAX_ALLOWED_ITERATION_PER_QUESTION = config.MAX_ALLOWED_ITERATION_PER_QUESTION
REPEAT_TIME_PER_QUESTION = config.REPEAT_TIME_PER_QUESTION
#a reset function to reset the sys.path
# sys.path = original_sys_path
final_res = []

def replace_names(text):
    text = re.sub(r'(?i)ChatGPT', 'Llama2', text)
    text = re.sub(r'(?i)OpenAI', 'Meta', text)
    return text
    
def process_raw_jailbreak_prompts(model_name,question_count):
    # file_path = '../../Data/data.csv'
    # jailbreak_path = './jailbreak-prompt.xlsx'
    openAI_model = False
    f = open(f'../DeepInception/res/data_main.json',) 
    datas = json.load(f) 
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
    
    if openAI_model:
        local_model = models.OpenAILLM(model_path)
    else:
        local_model = models.LocalVLLM(model_path=model_name_absolute, model_name=model_name)

    # df1 = pd.read_excel(jailbreak_path)
    # prompts_templates = df1['text']
    # train_data = pd.read_csv(file_path)
    # questions = train_data['goal'].tolist()
        
    final_results = []
    prompts = []
    for idx, data in enumerate(datas):
        question_bkp = data['questions']
        # if args.exp_name in ['main', 'further_q']:
        #     questions = [data['inception_attack']] + data['questions']
        # else:
        #     questions = data['questions']
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
    
    # for question in questions:
    #     # CURRENT_REPEAT = 0
    #     print(f"running {question}")
    #     # Accessing the 'goal' column for each row
    #     # if n == question_count:
    #     #     break

    #     # n += 1
        
    #     # while CURRENT_REPEAT < REPEAT_TIME_PER_QUESTION:
    #     #     CURRENT_REPEAT+= 1
    #     #     print(f"running {question} and repeat time is  {CURRENT_REPEAT}")
    #     prompts = []
    #     templates= []
    #     original_prompts = []
    #     # question_bkp = []
    #     counter = 0
    #     for template in prompts_templates:
    #         #skip first one, not suable
    #         # print(f"running {template}")
    #         if counter == 0:
    #             counter =1 
    #             continue
    #         prompt = template.replace('[INSERT PROMPT HERE]', question)
    #         prompt = replace_names(prompt)
    #         original_prompts.extend([prompt]*REPEAT_TIME_PER_QUESTION)
    #         prompts.append(local_model.create_conv_prompt(prompt))
    #         templates.extend([template]*REPEAT_TIME_PER_QUESTION)
    #         # question_bkp.extend([question]*REPEAT_TIME_PER_QUESTION)

    #     responses = local_model.generate_batch(prompts,n=REPEAT_TIME_PER_QUESTION,temperature=1)
    #     #flat this two dimensional list
    #     # responses = [item for sublist in responses for item in sublist]
    #     print(f"responses length are {len(responses)}")
    #     print(f"showing the first entry of responses {responses[0]}")
    #     i = 0
    #     for prompt,template,response in zip(original_prompts,templates,responses):
    #     # response = response[0].outputs[0].text
    #         i = i%REPEAT_TIME_PER_QUESTION + 1
    #         if i == 0:
    #             i = REPEAT_TIME_PER_QUESTION
    #         final_res.append({"template":template,"question":question,"prompt":prompt,"response":response,"iteration":i})
            

    # # # with open('processed_local_jailbreak_res_s.json', 'w') as f:
    # # #     json.dump(processed_local_res_s, f, indent=4)
    # # # with open('processed_local_jailbreak_res_f.json', 'w') as f:
    # # #     json.dump(processed_local_res_f, f, indent=4)
    # # path_name = model_name.replace("/", "_")
    # # if not os.path.exists(f"/content/llm_attack_arena/Attacks/TemplateJailbreak/Results"):
    # #     os.makedirs(f"/content/llm_attack_arena/Attacks/TemplateJailbreak/Results")
    # # print(json.dumps(final_res, indent=4))
    # # with open(f'/content/llm_attack_arena/Attacks/TemplateJailbreak/Results/TemplateJailbreak_{path_name}.json', 'w') as f:
    # #     json.dump(final_res, f, indent=4)
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