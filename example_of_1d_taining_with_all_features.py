import mydatasets
import mynetworks
from torch.utils.data import DataLoader
import torch

# Load dataset
# ------------
#specify batch size
batch_size = 4096

#specify file path
ncfile = "data/mera-sample-v1.2010-2011-123h.nc"

#The TemperatureLandcover1D class is used to create instances of the dataset for training and testing.
#The subset parameter is set to "train" for the training data and "test" for the test data.
#The normalize parameter is set to True for both datasets.
#The reduce_to parameter is set to 3,000,000 for both datasets.

training_data = mydatasets.FeaturesLandcover1D(
    ncfile, subset="train", normalize=True, reduce_to=3_000_000
)
test_data = mydatasets.FeaturesLandcover1D(
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
net = mynetworks.LinearRegressionWithEmbeddingAndFeatures()
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

          # Ensure baseline is compared to the t30m variable 
            baseline += loss_fn(x[:,0], y.squeeze()).item()

    test_loss /= num_batches
    print(
        f"Test Error: \n Mean squared error: {test_loss:>8f} (baseline: {baseline:>8f})\n"
    )


epochs = 20
for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    train(train_dataloader, net, loss_fn, optimizer)
    test(test_dataloader, net, loss_fn)
print("Done!")









""" Etracting weights for Feature Importance """

# extract the weights into a list
model_weights = net.linear.weight.data.squeeze().tolist()

# list of input variables
numeric_input = ['t30m', 't500hPa', 't850hPa', 'psurface', 'psea', 
                 'nlong', 'nshort',  'e30m', 'n30m', 'h2m', 'h30m', 'e850hPa', 
                 'n850hPa', 'orography', 'eastings', 'northings', 'landcover_embedding_0',
                'landcover_embedding_1','landcover_embedding_2']

weights = {}

#  weights for numeric input variables
for i, var in enumerate(numeric_input):
    weights[var] = model_weights[i]

# bar plot of the weights
plt.barh(numeric_input, model_weights)
plt.xlabel('Weights')
plt.ylabel('Features')
plt.tight_layout()

# Print out the weights
print("\n\n Features Weights:\n")
for variable, weight in weights.items():
    print(f"{variable}: {weight}")

# Print out weights less than 5% in asbsolute 
print("\n\n Features with weights< 5%:\n")
for variable, weight in weights.items():
    if abs(weight) < 0.05:
        print(f"{variable}: {weight}")
