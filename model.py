import logging
import math
import os
import string

import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = q @ einops.rearrange(k, '... seq_length embed_dim -> ... embed_dim seq_length')
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = attention @ v
    return values, attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def _attn(self, q, k, v, mask):
        return scaled_dot_product(q, k, v, mask=mask)

    def forward(self, x, mask=None, return_attention=False):
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = einops.rearrange(qkv, 'batch seq_length (num_heads head_dim) -> batch num_heads seq_length head_dim', head_dim=3*self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self._attn(q, k, v, mask=mask)
        values = einops.rearrange(values, 'batch num_heads seq_length head_dim -> batch seq_length (num_heads head_dim)')
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
    
    
class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out, _ = self.self_attn(x, mask=mask, return_attention=True)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x



class PositionalEncoding(nn.Module):
    """ From https://pytorch.org/tutorials/beginner/transformer_tutorial.html """
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        
        if batch_first:
            pe = torch.zeros(1, max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(0).unsqueeze(2)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class InductionTransformer(pl.LightningModule):
    def __init__(
        self, 
        layers=2, 
        width=128, 
        heads=4, 
        num_tokens=7, 
        max_seq_len=5, 
        dropout=0,
        batch_first=True,
        optim_kwargs=None, 
    ):
        
        super().__init__()
        self.save_hyperparameters()
        
        if not batch_first:
            raise NotImplementedError("Batch first not implemented yet.")
        
        # set optim kwargs
        if optim_kwargs is None:
            self.optim_kwargs = {
                'lr': 1e-3,
                'weight_decay':1,
                'betas': (0.9, 0.98),
            }
        else:
            self.optim_kwargs = optim_kwargs

        # input layer
        self.embedding = nn.Embedding(num_tokens, width)
        
        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model=width, max_len=max_seq_len, dropout=dropout, batch_first=batch_first)
        
        # transformer layers
        self.transformer = nn.Sequential(*[
            EncoderBlock(
                input_dim=width, 
                dim_feedforward=width*4,
                num_heads=heads, 
                dropout=dropout,
            ) for _ in range(layers)
        ])

        # output layer
        self.output = nn.Linear(width, num_tokens)
        
        # causal masking        
        self.register_buffer("self_attn_mask", torch.ones([max_seq_len, max_seq_len], device=self.device).tril())

        # loss
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        if self.hparams.batch_first:
            context_len = x.shape[1]
        else:
            context_len = x.shape[0]
        x = self.pos_encoding(self.embedding(x))
        for block in self.transformer:
            x = block(x, self.self_attn_mask[:context_len,:context_len])
        x = self.output(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x[:,:-1])
        loss = self.loss(y_hat[:,-1], x[:,-1])
        acc = (torch.argmax(y_hat[:,-1], dim=1) == x[:,-1]).sum() / x.shape[0]
        self.log('Training/Accuracy', acc, on_step=True)
        self.log('Training/Loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x[:,:-1])
        loss = self.loss(y_hat[:,-1], x[:,-1])
        acc = (torch.argmax(y_hat[:,-1], dim=1) == x[:,-1]).sum() / x.shape[0]
        self.log('Validation/Accuracy', acc, on_epoch=True)
        self.log('Validation/Loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), **self.optim_kwargs)
        # warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step/10, 1)) 
        # self.scheduler = warmup_scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=1e-5, T_max=15000)
        self.scheduler = cosine_scheduler
        return {
            'optimizer': self.optimizer, 
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'frequency': 1,
                'interval': 'step'
                }
        }

class InductionTokenizer:
    def __init__(self):
        self.allowed_tokens = list(string.ascii_lowercase)
    
    def tokenize(self, text):
        return list(text)
    
    def encode(self, text):
        text = self.tokenize(text)
        if len(text) != 5:
            raise ValueError("Text must be 5 characters long.")
        for letter in text:
            if letter not in self.allowed_tokens:
                raise ValueError("Invalid character in text: {}".format(letter))
        
        text = list(map(lambda x: self.allowed_tokens.index(x), text))
        return torch.from_numpy(np.array(text))[None]
    
    
        