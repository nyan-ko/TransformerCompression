import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.linalg as linalg
import transformers.models.llama.modeling_llama as llama
from transformers import LlamaTokenizer
from datasets import load_dataset

from slicegpt import load_sliced_model

from evaluation_hook import ForwardHook

import sys

normal_output = torch.load(sys.argv[1], weights_only=False)
sliced_output = torch.load(sys.argv[2], weights_only=False)

# print(normal_output.out[0].shape)
# print(sliced_output.out[0].shape)

out = []
batch = 8
for i in range(batch):
    error = linalg.matrix_norm(normal_output - sliced_output)
    rel_err = error / linalg.matrix_norm(normal_output)
    out.append(rel_err)
    print(rel_err)

torch.save(out, "out.pt")