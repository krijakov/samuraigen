#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

Description: Bigram based model using a fully fledged multi-headed attention system

'''

import torch
import torch.nn as nn
from torch.nn import functional as F

import math

from dataclasses import dataclass

@dataclass
class ModelConfigs:
    vocab_size: int = 81

    block_size: int = 6
    n_emb: int = 32
    n_heads: int = 4
    head_size: int = n_emb // n_heads

    batch_size: int = 8
    dropout_rate: float = 0.1

    train_rate: float = 0.9 

class SingleHeadAttention(nn.Module):
    """Single head of self-attention"""
    # init from a ModelConfigs class
    def __init__(self, config):
        super().__init__()

        assert config.n_emb % config.head_size == 0 # to ensure the ability to concatanate them later
        self.n_emb = config.n_emb
        self.head_size = config.head_size
        # init the linear layers:
        self.key = nn.Linear(config.n_emb, self.head_size, bias=False)
        self.value = nn.Linear(config.n_emb, self.head_size, bias=False)
        self.query = nn.Linear(config.n_emb, self.head_size, bias=False)

        self.register_buffer("causal", torch.ones(config.block_size, config.block_size))

        # dropout layer:
        self.dropout = nn.Dropout(config.dropout_rate)


    def forward(self, x):
        _, block, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))) 
        # casuality:
        att = att.masked_fill(self.causal[:block, :block] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        return y

class MultiHeadAttention(nn.Module):
    """Multiple single head modules in paralell then concatanate the result"""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_heads)])
        # extra linear layer and dropout:
        self.proj = nn.Linear(config.n_emb, config.n_emb)
        self.dropout = nn.Dropout(config.dropout_rate)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatanate alongside the last dimension
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """MLP module with ReLU nonlinearity"""
    def __init__(self, config):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(config.n_emb, 4*config.n_emb, bias=True), # why 4x? I guess just to have more parameters
            nn.ReLU(),
            nn.Linear(4*config.n_emb, config.n_emb), 
            nn.Dropout(config.dropout_rate),
        )

    def forward(self, x):
        return self.layers(x)
    
# expects a batch of embedded examples:
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_emb // config.n_heads
        # attention:
        self.att = MultiHeadAttention(config)
        # feed forward:
        self.ffwd = FeedForward(config)
        # layernorms:
        self.layernorm1 = nn.LayerNorm(config.n_emb)
        self.layernorm2 = nn.LayerNorm(config.n_emb)

    def forward(self, x):
        x = x + self.att(self.layernorm1(x)) # residual connection
        x = x + self.ffwd(self.layernorm2(x))
        return x
    
class NameGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # embedding:
        self.token_emb = nn.Embedding(config.vocab_size, config.n_emb)
        self.pos_emb = nn.Embedding(config.vocab_size, config.n_emb)
        # transformer:
        self.attn = Transformer(config) # add maybe more transformer layers
        self.ln_final = nn.LayerNorm(config.n_emb)
        # final linear layer:
        self.linear_final = nn.Linear(config.n_emb, config.vocab_size)

    def forward(self, inp, target = None): # input is a batch of tokenized examples of size (batch_size, block_size)
        batch, block = inp.shape
        # embedding:
        emb_tok = self.token_emb(inp)
        emb_pos = self.pos_emb(torch.arange(0, block)) 
        x = emb_pos + emb_tok
        # layers:
        x = self.attn(x)
        x = self.ln_final(x)
        logits = self.linear_final(x)
        # loss:
        if target is None:
            loss = None

        else:
            batch, block, vocab = logits.shape
            logits = logits.view(batch * block, vocab)
            target = target.view(batch * block)
            loss = F.cross_entropy(logits, target)

        return logits, loss
    

    def generate(self, curr_context, num_new_tokens):
        """Generates new tokens using the current state of the model and an input context"""
        for _ in range(num_new_tokens):
            # crop the current context to ensure (batch, block) shape, since we'll be adding to it:
            last_context = curr_context[:, -self.config.block_size:]
            # propagate throught the model:
            logits, _ = self(last_context)
            # focus only on the last step: this might change
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # sample the distribution:
            next_context = torch.multinomial(probs, num_samples=1)
            curr_context = torch.cat((curr_context, next_context), dim=1)
        return curr_context    