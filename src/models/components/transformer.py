# code adapted from lucidrains/vit-pytorch/vit.py (https://github.com/lucidrains/vit-pytorch)
import torch
from torch import nn
import hydra
from einops import rearrange, repeat
from .discrete_layers.abstract_discrete_layer import AbstractDiscreteLayer


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout = 0.):
        super().__init__()
        dim_head = dim // num_heads
        self.attn = PreNorm(dim, Attention(dim, heads=num_heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
            x = self.attn(x) + x
            x = self.ff(x) + x
            return x
    

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        dim_head = dim // heads
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    

class TransformerDBN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = kwargs
        # Add embedding layer
        self.token_embedding = nn.Embedding(self.hparams['num_embedding'], embedding_dim=self.hparams['embedding_dim'])
        self.pos_embedding = nn.Parameter(torch.randn(1, self.hparams['seq_len'] + 1, self.hparams['embedding_dim']))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hparams['embedding_dim']))
        self.dropout = nn.Dropout(self.hparams['emb_dropout'])
        
        self.layers = nn.ModuleList([])

        for i in range(self.hparams['depth']):
            transformer_layer = hydra.utils.instantiate(self.hparams['transformer_layer'], _recursive_=False)
            self.layers.append(transformer_layer)

            if self.hparams['dbn_after_each_layer'] or (i == self.hparams['depth'] - 1 and self.hparams['dbn_last_layer']):
                print("Creating a discrete bottleneck layer.")
                discrete_layer = hydra.utils.instantiate(self.hparams['discrete_layer'])
                if self.hparams['shared_embedding_dbn']:
                    discrete_layer.dictionary = self.token_embedding
                self.layers.append(discrete_layer)
        
        self.pool = self.hparams['pool']
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.hparams['embedding_dim']),
            nn.Linear(self.hparams['embedding_dim'], self.hparams['output_dim'])
        )

    def forward(self, inputs):
        inputs = inputs.int()
        x = self.token_embedding(inputs)
        b, n, _ = x.shape
        bottleneck_loss = 0

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)

        for layer in self.layers:
            if isinstance(layer, AbstractDiscreteLayer):
                indices, probs, x, disc_loss = layer(x, supervision=self.hparams['supervision']) # TODO: for now I'm adding this to the config file...
                bottleneck_loss += disc_loss
            else:
                x = layer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x), bottleneck_loss


class TokenTransformer(nn.Module):
    def __init__(self, *, seq_len, output_dim, dim, depth, heads, mlp_dim, pool='cls', 
            dim_head=64, dropout=0., emb_dropout=0., inputs_are_pos_neg_ones=True, albert=False):
        super().__init__()

        self._inputs_are_pos_neg_ones = inputs_are_pos_neg_ones
        self.token_embedding = nn.Embedding(num_embeddings=2, embedding_dim=dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.albert = albert

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim)
        )

    def forward(self, inputs):
        if self._inputs_are_pos_neg_ones:
            # convert from +1/-1 to 0/1
            inputs = (inputs + 1) / 2
        inputs = inputs.int()
        x = self.token_embedding(inputs)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)