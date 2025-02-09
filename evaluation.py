import torch
import torch.nn as nn
import transformers.models.llama.modeling_llama as llama
from transformers import LlamaTokenizer
from datasets import load_dataset


from evaluation_hook import ForwardHook

wiki = load_dataset("wikipedia", "20200501.en")

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(wiki["text"], return_tensors="pt", padding=True, truncation=True).to("cuda")

with torch.no_grad():
    sliced_model(**inputs)

torch.save(sliced_hook, "hook_output.pt")

sliced_hook.close()