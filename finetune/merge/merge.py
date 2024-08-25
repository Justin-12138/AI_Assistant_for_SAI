from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

base_model_name = "/home/lz/Desktop/llama.cpp/testmodels/llama3.1-8b/hf"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

lora_model_name = "/home/lz/Downloads/lora"
peft_config = PeftConfig.from_pretrained(lora_model_name)
lora_model = PeftModel.from_pretrained(base_model, peft_config)


tokenizer = AutoTokenizer.from_pretrained(base_model_name)


lora_model.merge_and_unload()

save_path = "./merged_model"
lora_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
