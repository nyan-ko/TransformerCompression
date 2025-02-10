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

print("normal: " + str(len(normal_output.out)))
print("sliced: " + str(len(sliced_output.out)))

n = min(len(normal_output.out), len(sliced_output.out))

out = []
batch = 8
for i in range(n):
    for j in range(batch):
        print(normal_output.out[i][j].shape)
        print(sliced_output.out[i][j].shape)
        error = linalg.matrix_norm(normal_output.out[i][j] - sliced_output.out[i][j])
        rel_err = error / linalg.matrix_norm(normal_output.out[i][j])
        out.append(rel_err)
        print(rel_err)

torch.save(out, "out.pt")