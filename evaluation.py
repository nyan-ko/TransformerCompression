import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers.models.llama.modeling_llama as llama
from transformers import LlamaTokenizer
from datasets import load_dataset

from slicegpt import load_sliced_model

from evaluation_hook import ForwardHook

import sys

wiki = load_dataset("wikipedia", "20220301.en", split="train")
wiki = wiki.train_test_split(test_size=0.005, seed=42)["test"]

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = wiki.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

loader = DataLoader(tokenized_dataset, batch_size=8)

device = torch.device("cuda")

model_adapter, _ = load_sliced_model("meta-llama/Llama-2-7b-hf", sys.argv[1], sparsity=sys.argv[2], token=sys.argv[3])
model = model_adapter.model.to(device)
sliced_hook = ForwardHook(model)
model.eval()

for batch in loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

torch.save(sliced_hook, "hook_output.pt")
sliced_hook.close()