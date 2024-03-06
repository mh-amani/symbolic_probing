# code adapted from lucidrains/vit-pytorch/vit.py (https://github.com/lucidrains/vit-pytorch)
import torch
from torch import nn
import hydra
from einops import rearrange, repeat
from .discrete_layers.abstract_discrete_layer import AbstractDiscreteLayer
from transformers import GPT2LMHeadModel, GPT2Config


class GPT2Classifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = kwargs
        # Add embedding layer
        # initializing gpt2 model at random
        self.gpt2 = GPT2LMHeadModel(GPT2Config(**self.hparams['gpt2_config']))
                
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.hparams['embedding_dim']),
            nn.Linear(self.hparams['embedding_dim'], self.hparams['output_dim'])
        )

    def forward(self, inputs):
        x = inputs.int()
        # get embeddings from gpt2 model
        x = self.gpt2(input_ids=x)['hidden_states']
        # get the last hidden state
        x = x[-1][:, -1, :]
        # pass through mlp head        
        return self.mlp_head(x)

