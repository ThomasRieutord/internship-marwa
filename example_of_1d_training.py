#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Marwa Mahmood intership (UCD student, May-July 2024)

Example code to call a 1D dataset, a simple network and train it
"""

import mydatasets
import mynetworks
from torch.utils.data import DataLoader
import torch

# Load dataset
# ------------
batch_size = 4096
ncfile = "data/mera-sample-v1.2010-2011-123h.nc"
training_data = mydatasets.TemperatureLandcover1D(
    ncfile, subset="train", normalize=True, reduce_to=3_000_000
)
test_data = mydatasets.TemperatureLandcover1D(
    ncfile, subset="test", normalize=True, reduce_to=3_000_000
)

x0, c0, y0 = training_data[0]
print(f"The dataset has {len(training_data)} items. Each item looks like {x0, c0, y0}")


train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

x, c, y = next(iter(train_dataloader))
print(f"The dataloader create batches of items of shape {x.shape, c.shape, y.shape}")


# Network
# -------
net = mynetworks.LinearRegressionWithEmbedding()
y_pred = net(x, c)


# Loss and optimizer
# -------------------
loss_fn = torch.nn.MSELoss()
print(f"Loss value: {loss_fn(y_pred, y)}")

optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)


# Training
# --------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    for batch, (x, c, y) in enumerate(dataloader):
        x, c, y = x.to(device), c.to(device), y.to(device)

        # Compute prediction error
        pred = model(x, c)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    baseline = 0
    with torch.no_grad():
        for x, c, y in dataloader:
            x, c, y = x.to(device), c.to(device), y.to(device)
            pred = model(x, c)
            test_loss += loss_fn(pred, y).item()
            baseline += loss_fn(x, y).item()

    test_loss /= num_batches
    print(
        f"Test Error: \n Mean squared error: {test_loss:>8f} (baseline: {baseline:>8f})\n"
    )


epochs = 2
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    train(train_dataloader, net, loss_fn, optimizer)
    test(test_dataloader, net, loss_fn)
print("Done!")
