#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Marwa Mahmood intership (UCD student, May-July 2024)

Pytorch datasets
"""
import torch
from torch import nn



""" Simple Network """

class LinearRegressionWithEmbedding(nn.Module):
    def __init__(self, n_landcovers=33, embedding_size=3, output_size=1):
        super().__init__()
        self.landcover_embedding = nn.Embedding(n_landcovers, embedding_size)
        self.linear = nn.Linear(embedding_size + 1, output_size)

    def forward(self, x, c):
        emb = self.landcover_embedding(c)
        x = torch.cat([x, emb], dim=1)
        return self.linear(x)



""" Simple Network with 1 Hidden Layer """

class LinearRegressionWithEmbeddingHiddenLayer(nn.Module):
    def __init__(self, n_landcovers=33, embedding_size=3, output_size=1):
        super().__init__()
        self.landcover_embedding = nn.Embedding(n_landcovers, embedding_size)
        self.fc1 = nn.Linear(embedding_size+1, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x, c):
        emb = self.landcover_embedding(c)
        x = torch.cat([x, emb], dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)



""" Network with all features """

class LinearRegressionWithEmbeddingAndFeatures(nn.Module):
    def __init__(self, n_landcovers=33, embedding_size=3, input_size=14, output_size=1):
        super().__init__()
        self.landcover_embedding = nn.Embedding(n_landcovers, embedding_size)
        self.linear = nn.Linear(embedding_size + input_size, output_size)

    def forward(self, x, c):
        emb = self.landcover_embedding(c)
        
        #ensure the input is type float
        x = torch.cat([x.to(torch.float32), emb], dim=1)
        return self.linear(x)
    


""" Network with 1 Hidden Layer and all features """

class LinearRegressionWithEmbeddingHiddenLayerAndFeatures(nn.Module):
    def __init__(self, n_landcovers=33, embedding_size=3, input_size=14, output_size=1):
        super().__init__()
        self.landcover_embedding = nn.Embedding(n_landcovers, embedding_size)
        self.fc1 = nn.Linear(embedding_size + input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x, c):
        emb = self.landcover_embedding(c)
        
        #ensure the input is type float
        x = torch.cat([x.to(torch.float32), emb], dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
