import os
from huggingface_hub import login as huglogin
#huglogin("Add_Your_Token_Here")

model_names_list = {
    'phi2':"microsoft/phi-2",
    'llama':"meta-llama/Llama-3.2-1B",
    'deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
}

developers = {
    'phi2':"microsoft",
    'llama':"meta",
    'deepseek': 'deepseek-ai'
}

def get_model_path(model_name):
    return os.path.join(os.path.dirname(__file__), 'models', model_name)

def get_developer(model_name):
    if model_name in developers:
        return developers[model_name]
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available models are {list(developers.keys())}.")
    
def get_model_path_and_template(model_name):
    full_model_dict={
        "llama":{
            "path": "meta-llama/Llama-3.2-1B",
            "template":"llama-2"
        },
        "phi2":{
            "path": "microsoft/phi-2",
            "template": "llama-2"
        },
        "deepseek":{
            "path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "template": "llama-2"
        }
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template
    