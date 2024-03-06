from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from torch.nn import LayerNorm
class AbstractDiscreteLayer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()      
        
        self.vocab_size = kwargs['vocab_size']
        self.dictionary_dim = kwargs['dictionary_dim']

        self.temperature = kwargs.get('temperature', 1)
        self.label_smoothing_scale = kwargs.get('label_smoothing_scale', 0.001)
        
        self.out_layer_norm = LayerNorm(self.dictionary_dim)

        self.dictionary = nn.Embedding(self.vocab_size, self.dictionary_dim)
    
    def project_matrix(self,x,**kwargs):
        return x

    def project_embedding_matrix(self):
        self.dictionary.weight = torch.nn.Parameter(self.project_matrix(self.dictionary.weight))
    
    def forward(self, x,**kwargs):
        # scores are between 0 and 1, and sum to 1 over the vocab dimension.
        id, score, quantized_vector, quantization_loss  = self.discretize(x)
        return id, score, quantized_vector, quantization_loss

    @abstractmethod
    def discretize(self, x,**kwargs) -> dict:
        pass


    