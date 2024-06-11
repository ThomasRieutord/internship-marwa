#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Marwa Mahmood intership (UCD student, May-July 2024)

Pytorch datasets
"""
import torch
from torch import nn


class LinearRegressionWithEmbedding(nn.Module):
    def __init__(self, n_landcovers=33, embedding_size=3, output_size=1):
        super().__init__()
        self.landcover_embedding = nn.Embedding(n_landcovers, embedding_size)
        self.linear = nn.Linear(embedding_size + 1, output_size)

    def forward(self, x, c):
        emb = self.landcover_embedding(c)
        x = torch.cat([x, emb], dim=1)
        return self.linear(x)
