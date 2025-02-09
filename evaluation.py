import torch
import torch.nn as nn
import transformers.models.llama.modeling_llama as llama
from transformers import LlamaTokenizer
from datasets import load_dataset

from slicegpt import load_sliced_model

from evaluation_hook import ForwardHook

import sys

wiki = load_dataset("wikipedia", "20220301.en")

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(wiki["text"], return_tensors="pt", padding=True, truncation=True).to("cuda")

model_adapter, _ = load_sliced_model("meta-llama/Llama-2-7b-hf", sys.argv[1], sparsity=sys.argv[2], token=sys.argv[3])
sliced_hook = ForwardHook(model_adapter.model)

with torch.no_grad():
    model_adapter.model(**inputs)

torch.save(sliced_hook, "hook_output.pt")

sliced_hook.close()