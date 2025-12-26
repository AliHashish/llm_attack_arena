from transformers import AutoTokenizer
from fastchat.model import get_conversation_template
from vllm import LLM as VLLM
from vllm import SamplingParams

MAX_LEN = 2048

def truncate_prompts(tokenizer, prompts, max_len):
    # Truncate prompts to fit within max_len tokens
    truncated = []
    for prompt in prompts:
        tokens = tokenizer(prompt, truncation=True, max_length=max_len, return_tensors="pt")["input_ids"]
        truncated_prompt = tokenizer.decode(tokens[0], skip_special_tokens=True)
        truncated.append(truncated_prompt)
    return truncated

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class LocalVLLM(LLM):
    def __init__(self,
                 model_name,
                 model_path,
                 gpu_memory_utilization=0.90,
                 system_message=None
                 ):
        super().__init__()

        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Initialize VLLM model based on model_path
        if 'deepseek' in model_path or 'llama' in model_path:
            self.model = VLLM(model=model_name, dtype="float16",
                # tensor_parallel_size=1,   # num of GPUs to run on
                max_model_len=110072,
                enable_chunked_prefill=False,   # <-- important
                gpu_memory_utilization=gpu_memory_utilization)
        elif 'Qwen' in model_path:
            self.model = VLLM(model=model_name, dtype="float16",
                # tensor_parallel_size=1,   # num of GPUs to run on
                max_model_len=30000,
                enable_chunked_prefill=False,   # <-- important
                gpu_memory_utilization=gpu_memory_utilization)
        else:
            self.model = VLLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization)
        
        if system_message is None and 'llama' in model_path:
            # system message for the llama model
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    def set_system_message(self, conv_temp):
        # Set system message in conversation template
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, temperature=1.0, max_tokens=150,top_p=1,top_k=-1, repetition_penalty=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1):
        # Single prompt generation
        prompts= [prompt]
        if n > 1:
            result=self.generate_batch(prompts,temperature,max_tokens,top_p,top_k,repetition_penalty,frequency_penalty,presence_penalty,n=n)
        else:
            result=self.generate_batch(prompts,temperature,max_tokens,top_p,top_k,repetition_penalty,frequency_penalty,presence_penalty,n=n)[0]
        return result

    def generate_batch(self, prompts, temperature=1.0, max_tokens=150,top_p=1,top_k=-1, repetition_penalty=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1):
        # Batch prompt generation
        prompts = truncate_prompts(self.tokenizer, prompts, MAX_LEN)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,top_p=top_p,top_k=top_k,repetition_penalty=repetition_penalty,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,n=n)
        results = self.model.generate(prompts, sampling_params, use_tqdm=False)
        outputs = []
        for result in results:
            for output in result.outputs:  # Iterate over all outputs in each result
                generated_text = output.text
                outputs.append(generated_text)
        return outputs

    def create_conv_prompt(self,prompt):
        # Create conversation prompt using the conversation template
        conv_template = get_conversation_template(
            self.model_name
        )
        conv_template.messages = [] 
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        # System message, prompt, response

        full_prompt = conv_template.get_prompt()
        return full_prompt