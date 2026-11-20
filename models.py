import os
from directory_config import *

import transformers, torch
from pathlib import Path
from openai import OpenAI
transformers.utils.logging.disable_progress_bar()
transformers.utils.logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    print(f"using device {device}, {torch.cuda.device_count()} GPUs {torch.cuda.get_device_name(0)}")
except:
    print(f"using device {device}")

class Model:
    def __init__(self, model_type: str, temperature:float=0):
        self.token_usage = 0
        self.total_cost = 0
        self.model_type = model_type
        self.model_string = "model string missing"
        self.device=device
        self.temperature=temperature
        self.do_sample = temperature > 0


        if model_type == "llama":
            self.model_string = "meta-llama/Llama-3.1-8B-Instruct" #"meta-llama/Llama-3.3-70B-Instruct"
            # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
            # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B") 
            # if tokenizer.pad_token is None:
            #     tokenizer.pad_token = tokenizer.eos_token
            self.pipeline = transformers.pipeline(
                    "text-generation", model=self.model_string, model_kwargs={"dtype": torch.bfloat16}, 
                    device=device, return_full_text=False, max_length=None, do_sample = False, temperature=self.temperature)
            self.tokenizer = self.pipeline.tokenizer
            self.model = self.pipeline.model
        if model_type == "llama-big":
            self.model_string = "meta-llama/Llama-3.3-70B-Instruct"
            self.pipeline = transformers.pipeline(
                    "text-generation", device_map="auto", model=self.model_string, model_kwargs={"dtype": torch.bfloat16}, 
                    return_full_text=False, max_length=None, do_sample = False, temperature=self.temperature)
            
        if model_type == "gpt-4o" or model_type == "gpt-5.4":
            self.model_string = model_type
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        if model_type == "qwen":
            self.model_string = "Qwen/Qwen3.5-9B" # -2507" #"Qwen/Qwen3-30B-A3B-Instruct-2507"
            self.pipeline = transformers.pipeline("text-generation", torch_dtype=torch.bfloat16, model=self.model_string, device=device, return_full_text=False, 
                                                  max_length=None, temperature=self.temperature, do_sample=self.do_sample)
            self.tokenizer = self.pipeline.tokenizer
            self.model = self.pipeline.model
  
        if self.temperature != 0:
            self.model_string += f" at temperature {self.temperature}"
        
        
    def prompt_model(self, prompt, max_tokens=100, steering=False):
        if type(prompt) == list: #already in chat template form
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        if self.model_type in ["llama", "llama-big"]:
            if steering:
                messages.insert(0, {"role": "system", "content": f"In addition to your expertise specified below, your political opinions are: {steering}"})
            #chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            output = self.pipeline(messages, max_new_tokens=max_tokens, max_length=None, pad_token_id=self.pipeline.tokenizer.eos_token_id)[0]['generated_text']
            try:
                self.token_usage += len(self.pipeline.tokenizer(self.pipeline.tokenizer.apply_chat_template(messages), tokenize=False)) + len(self.pipeline.tokenizer(output))
            except ValueError:
                pass
                #print("WARNING: Couldn't calculate token usage due to ValueError")
            return output
        if self.model_type == "qwen":
            if steering:
                messages.insert(0, {"role": "system", "content": f"In addition to your expertise spcified below, your political opinions are: {steering}"})
            chat_prompt = self.tokenizer.apply_chat_template(messages, enable_thinking=False, tokenize=False, add_generation_prompt=True)
            output =  self.pipeline(chat_prompt, max_new_tokens=max_tokens, max_length=None, pad_token_id=self.pipeline.tokenizer.eos_token_id)[0]['generated_text'].strip()
            self.token_usage += len(self.tokenizer(chat_prompt)) + len(self.tokenizer(output))
            return output
        if self.model_type == "gpt-4o" or self.model_type == "gpt-5.4":
            if steering:
                messages.insert(0, {"role": "system", "content": f"{steering}"})
            response = self.client.chat.completions.create(model=self.model_type,messages=messages,temperature=self.temperature, frequency_penalty=0.0, seed=15)
            response_text = response.choices[0].message.content
            self.token_usage += response.usage.total_tokens
            self.total_cost += 2.5 * response.usage.prompt_tokens / 1000000 + 10 * response.usage.completion_tokens / 1000000
            return response_text

