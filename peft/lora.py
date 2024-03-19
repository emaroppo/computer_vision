import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-v0.1-GGUF", load_in_8_bits=True
)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-v0.1-GGUF")

for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        param.data.to(torch.float32)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


config = LoraConfig.from_pretrained(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
