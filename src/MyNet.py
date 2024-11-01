import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import ast
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, att_weight = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src,att_weight

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class LinearDecoder(nn.Module):
    def __init__(self,d_model,n_class,dropout):
        super(LinearDecoder, self).__init__()
        self.decoder1 = nn.Linear(d_model, d_model // 2)
        self.decoder2 = nn.Linear(d_model // 2, d_model // 4)
        self.classifier = nn.Linear(d_model // 4, n_class)

        self.norm1 = nn.LayerNorm(d_model // 2)
        self.norm2 = nn.LayerNorm(d_model // 4)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.norm1(self.dropout(self.leakyrelu(self.decoder1(x))))
        x = self.norm2(self.dropout(self.leakyrelu(self.decoder2(x))))
        x = self.classifier(x)
        return x


class MyModel(nn.Module):
    def __init__(self, d_model, dropout, n_class, vocab_size,nlayers,nhead,dim_feedforward):
        super(MyModel, self).__init__()
        self.embeding = nn.Embedding(vocab_size,d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer_encoder = []
        for i in range(nlayers):
            self.transformer_encoder.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout))
        self.transformer_encoder= nn.ModuleList(self.transformer_encoder)

        self.decoder = LinearDecoder(d_model+dim_feedforward,dropout,n_class)


    def forward(self, x1,x2): #x2 - seq
        x1 = x1.to(torch.float32)

        x2 = self.embeding(x2)
        x2 = x2.permute(1,0,2)

        attention_weights = []
        for layer in self.transformer_encoder:
            x2,attention_weights_layer=layer(x2)
            attention_weights.append(attention_weights_layer)
        attention_weights=torch.stack(attention_weights)

        # (sequence_length, batch_size, features)->(batch_size, sequence_length, features)
        x2 = x2.permute(1,0,2)
        x2 = torch.mean(x2, dim=1)
        x = torch.cat((x1, x2), dim=1)

        x = self.decoder(x)

        return x