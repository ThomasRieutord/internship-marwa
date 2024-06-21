class TemperatureLandcover1D(Dataset):
    def __init__(self, path, subset="train", seed=87, normalize=False, reduce_to=None):
        self.path = path
        self.subset = subset
        self.seed = seed

        # Load the data
        ds = xr.open_dataset(self.path)
        t2m = ds.air_temperature_at_2_metres.values  # (143, 489, 529)
        t30m = ds.air_temperature_at_30_metres.values  # (143, 489, 529)
        landcover = ds.landcover.values  # (member, x, y) int8

        
        
        # To include all categories, get the unique classes from the int68 landcover 
        landcover_classes = np.unique(landcover)
      
        # Create a dictionary to map landcover categories to integers
        # enumerate(landcover_classes) paires each unique class (note this is in integer since landcover type is int64) in landcover with a unique integer (levels)
        # the full code then maps the classes with their corresponding unique integer 
        landcover_to_ix = {category: i for i, category in enumerate(landcover_classes)}
        

        # Flatten the data
        t2m = t2m.flatten()
        t30m = t30m.flatten()

        # Replace the landcover variable categories to an array of integers and flatten the array
        # create a comprehension list that replaces each category in the 1D flattended landcover with its corresponding unique integer from landcaover_to_ix 
        # then turn this into an array
        landcover = np.array([landcover_to_ix[category] for category in landcover.flatten()])
        
        
        # Normalize the temperature data
        if normalize:
            t2m = (t2m - t2m.mean()) / t2m.std()
            t30m = (t30m - t30m.mean()) / t30m.std()

        # Split the data into training, testing, and validation subsets
        np.random.seed(self.seed)
        if reduce_to is None:
            reduce_to = t2m.size

        train_idxs = np.random.randint(0, t2m.size, reduce_to // 2)  # 50% in training
        test_idxs = np.random.randint(0, t2m.size, reduce_to // 4)  # 25% in test
        val_idxs = np.random.randint(0, t2m.size, reduce_to // 4)  # 25% in validation

        if self.subset == "train":
            idxs = train_idxs
        elif self.subset == "test":
            idxs = test_idxs
        elif self.subset == "val":
            idxs = val_idxs
        else:
            raise ValueError("Unknown subset")
            

        # Convert the selected indices to tensors
        # use modulus operator (%) to ensure that the index (idxs) from train/test/validation is within the range of the t2m/t30m/landcovver array to avoid IndexError
        self.t2m = torch.tensor(t2m[idxs % t2m.size])
        self.t30m = torch.tensor(t30m[idxs % t30m.size])
        self.landcover = torch.tensor(landcover[idxs % landcover.size])

    def __len__(self):
        return len(self.t2m)

    def __getitem__(self, idx):
        """Return an item of the dataset.

        Parameters
        ----------
        idx: int
            Index of the item to return

        Returns
        -------
        x: float tensor of shape (1,)
            Temperature at 2m
        c: int64
            Landcover category
        y: float tensor of shape (1,)
            Temperature at 30m
        """
        x = self.t30m[idx].unsqueeze(0)
        c = self.landcover[idx].long()
        y = self.t2m[idx].unsqueeze(0)
        return x, c, y
