import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers.models.llama.modeling_llama as llama
from transformers import LlamaTokenizer
from datasets import load_dataset

from slicegpt import load_sliced_model

from evaluation_hook import ForwardHook

import sys

normal_output = torch.load(sys.argv[1])
sliced_output = torch.load(sys.argv[2])

print(normal_output.shape)
print(sliced_output.shape)
