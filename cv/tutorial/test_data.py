import torch
import os
import torchvision
import pandas as pd
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "/home/wushan/Desktop/data/banana-detection"

def load_data(is_train, path=DATA_DIR):
    ds_dir = "bananas_train" if is_train else "bananas_val"
    data = pd.read_csv(os.path.join(path, ds_dir, "label.csv"))
    data = data.set_index("img_name")

    X, Y = [], []
    for label, row in data.iterrows():
        img = os.path.join(path, ds_dir,"images",label)
        src = torchvision.io.read_image(img)
        tgt = row.tolist()
        X.append(src)
        Y.append(torch.tensor(tgt).long())
    X = torch.stack(X)
    Y= torch.stack(Y)/255 # a trick: class label = 0.
    
    return X, Y

class BananaDataset(Dataset):
    def __init__(self, is_train):
        self.X, self.Y = load_data(is_train)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index],self.Y[index]

def get_loader(is_train, batch_size):
    dataset = BananaDataset(is_train)
    return DataLoader(dataset, batch_size, True)

def get_test_data(batch_size):
    return get_loader(True, batch_size), get_loader(False, batch_size)