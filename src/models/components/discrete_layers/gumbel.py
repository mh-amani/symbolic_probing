from .abstract_discrete_layer import AbstractDiscreteLayer
import torch
from torch.nn.functional import gumbel_softmax


class GumbelDiscreteLayer(AbstractDiscreteLayer):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__(dims, **kwargs)
        self.hard = kwargs['hard'] # if True, use argmax in forward pass, else use gumbel softmax. the backwardpass is the same in both cases
        self.output_embedding = torch.nn.Linear(self.output_dim, self.vocab_size)

    def discretize(self, x,**kwargs) -> dict:
        score = gumbel_softmax(x, tau=self.temperature, hard=self.hard, dim=-1)
        x_quantized = torch.matmul(score, self.dictionary.weight)
        id = torch.argmax(score, dim=-1)
        quantization_loss = 0
        return id, score, x_quantized, quantization_loss
    