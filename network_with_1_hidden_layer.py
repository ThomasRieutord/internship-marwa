class LinearRegressionWithEmbedding(nn.Module):
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
