import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers.models.llama.modeling_llama as llama
from transformers import LlamaTokenizer
from datasets import load_dataset

from slicegpt import load_sliced_model

from evaluation_hook import ForwardHook

import sys

normal_output = torch.load(sys.argv[1], weights_only=False)
sliced_output = torch.load(sys.argv[2], weights_only=False)

print(normal_output.out[0].shape)
print(sliced_output.out[0].shape)
