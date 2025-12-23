
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
from config import (TARGET_TEMP, TARGET_TOP_P)
from language_models import HuggingFace
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import models
from utils import get_model_path_and_template

def load_attack_and_target_models(args):
    targetLM = TargetLM(model_name = args.target_model, 
                        max_n_tokens = args.target_max_n_tokens,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        preloaded_model = None,
                        )
    return targetLM


class TargetLM():
    """
        Base class for target language models.
        
        Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
            model_name: str, 
            max_n_tokens: int, 
            temperature: float,
            top_p: float,
            preloaded_model: object = None):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, prompts_list, defense_type):
        conv = common.conv_template(self.template)

        conv.append_message(conv.roles[0], prompts_list)
        conv.append_message(conv.roles[1], None)
        
        if 'gpt' in self.model_name:
            full_prompts = [conv.to_openai_api_messages()]
        else:
            full_prompts = conv.get_prompt() 
        
        if 'llama' in self.model_name:
            outputs_list = self.model.generate_batch(full_prompts,
                                                        max_tokens = self.max_n_tokens,
                                                        temperature = self.temperature,
                                                        top_p = self.top_p)
        else:
            outputs_list = self.model.batched_generate(full_prompts, 
                                                            max_n_tokens = self.max_n_tokens,  
                                                            temperature = self.temperature,
                                                            top_p = self.top_p
                                                        )
        return outputs_list



def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)
    print(f"\n\n\nmodelPath: {model_path}\n\n\n")
    model_name_absolute = "/".join(model_path.split("/")[-2:])
    model = AutoModelForCausalLM.from_pretrained(
            model_name_absolute, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True).eval().to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False
    ) 

    if 'llama' in model_path.lower():
        #use vllm
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
        local_model = models.LocalVLLM(model_path=model_name_absolute, model_name=model_name_absolute)
        return local_model, template
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    lm = HuggingFace(model_name, model, tokenizer)
    
    return lm, template