import torch
import torch.nn as nn
import transformers.models.llama.modeling_llama as llama
from transformers import LlamaTokenizer
from datasets import load_dataset

from evaluation_hook import ForwardHook

normal_model = llama.LlamaForCausalLM(llama.LlamaConfig())
normal_model.from_pretrained("llama")
normal_hook = ForwardHook(normal_model)
normal_model.eval()

sliced_model = llama.LlamaForCausalLM(llama.LlamaConfig())
sliced_model.load_state_dict(torch.load("out.pth"))
sliced_hook = ForwardHook(sliced_model)
sliced_model.eval()

wiki = load_dataset("wikipedia", "20200501.en")

tokenizer = LlamaTokenizer.from_pretrained("llama")
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(wiki["text"], return_tensors="pt", padding=True, truncation=True).to("cuda")

with torch.no_grad():
    normal_model(**inputs)
    sliced_model(**inputs)

for normal_out, sliced_out in zip(normal_hook.out, sliced_hook.out):
    print(f"normal: {normal_out}")
    print(f"sliced: {sliced_out}")
normal_hook.close()
sliced_hook.close()