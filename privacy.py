import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from matplotlib.dates import datestr2num
# plt.ion()

# Define all script inputs
FILENAME = 'augmented_data.csv'

userID = {
'a841f74e620f74ec443b7a25d7569545':0,
'22223276ea84bbce3a62073c164391fd':1,
'510635002cb29804d54bff664cab52be':2,
'7cbc37da05801d46e7d80c3b99fd5adb':3,
'7023889b4439d2c02977ba152d6f4c6e':4,
'8425a81da55ec16b7f9f80c139c235a2':5,
'6882f6cf8c72d6324ba7e6bb42c9c7c2':6,
'1e33db5d2be36268b944359fbdbdad21':7,
'892d2c3aae6e51f23bf8666c2314b52f':8,
}

class ChaniaDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.augmented_data = pd.read_csv(csv_file, header=0, usecols = [0]+[j for j in range(2,26)], converters={0:lambda x: datestr2num(x)})
        self.userlabels = pd.read_csv(csv_file, header=0, usecols=["iPhoneUID"], converters={"iPhoneUID": lambda x: userID[x]})
        self.transform = transform

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        data = self.augmented_data.iloc[idx].values
        data = data.astype('float').reshape(-1,25)
        user = self.userlabels.iloc[idx].values
        user = user.astype('int').reshape(-1,1)
        sample = {'x':data, 'u':user}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        data, user = sample['x'], sample['u']
        return {'x':torch.from_numpy(data), 'u':torch.from_numpy(user)}

chania_dataset = ChaniaDataset(csv_file=FILENAME, transform=ToTensor())

dataloader = DataLoader(chania_dataset, batch_size=16, shuffle=True)

def show_batch_wlabels(sample_batch):
    data, users = sample_batch['x'], sample_batch['u']
    batch_size = len(data)
    colors = users.numpy()[:,0].tolist()
    for i in range(batch_size):
        plt.scatter(data[:,:,13].numpy(), data[:,:,12].numpy(), c=colors)

for i_batch, sample_batch in enumerate(dataloader):
    if i_batch == 3:
        plt.figure()
        show_batch_wlabels(sample_batch)
        plt.show()
        break
