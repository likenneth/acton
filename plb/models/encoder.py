import math

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

inter_dim_dict = {2048: 512, 1024:256, 512: 128, 256: 128, 128: 64, 64: 64}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, tr_layer=6, tr_dim=512, j=51):
        super(Transformer, self).__init__()
        self.layers = tr_layer
        self.d_model = tr_dim
        self.pos_encoder = PositionalEncoding(self.d_model, 0)
        self.embedder = nn.Sequential(nn.Linear(j, inter_dim_dict[self.d_model]), nn.Linear(inter_dim_dict[self.d_model], self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, tr_layer)  # [L, B, F] --> [L, B, F]

    def forward(self, point_bank, len):
        # point_bank, [N, T, 51]
        max_len = point_bank.size(1)  # collate fuction collate across GPUs, so the padded length might not larger than the max in this GPU
        points = self.embedder(point_bank) * math.sqrt(self.d_model)
        points = self.pos_encoder(points.transpose(0, 1))  # [T, N, 512], transpose should be before pos_encoder
        mask = torch.stack([(torch.arange(max_len, device=len.device) >= _) for _ in len]).to(points.device)
        points = self.encoder(points, src_key_padding_mask=mask)  # [T, N, 512]
        # points = points[:, 0]  # use the first token as a summary of whole
        return points

class Transformer_wote(nn.Module):
    def __init__(self, tr_layer=6, tr_dim=512, j=51):
        super(Transformer_wote, self).__init__()
        self.layers = tr_layer
        self.d_model = tr_dim
        # self.pos_encoder = PositionalEncoding(self.d_model, 0)
        self.embedder = nn.Sequential(nn.Linear(j, inter_dim_dict[self.d_model]), nn.Linear(inter_dim_dict[self.d_model], self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, tr_layer)  # [L, B, F] --> [L, B, F]

    def forward(self, point_bank, len):
        # point_bank, [N, T, 51]
        max_len = point_bank.size(1)  # collate fuction collate across GPUs, so the padded length might not larger than the max in this GPU
        points = self.embedder(point_bank) * math.sqrt(self.d_model)
        points = points.transpose(0, 1)  # [T, N, 512], transpose should be before pos_encoder
        mask = torch.stack([(torch.arange(max_len, device=len.device) >= _) for _ in len]).to(points.device)
        points = self.encoder(points, src_key_padding_mask=mask)  # [T, N, 512]
        # points = points[:, 0]  # use the first token as a summary of whole
        return points
