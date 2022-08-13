from pathlib import Path
import pandas as pd
import os
import numpy as np
import torch
#from PIL import Image
from torch.utils.data import Dataset
from random import sample


class PCAP(Dataset):
    def __init__(self, data_root):
        self.data = []
        self.label = []
        allData = [[] for i in range(10)]
        allLabel = [[] for i in range(10)]
        print(np.array(allData).shape)
        for fi in os.listdir(data_root):
            if fi.endswith('.npz'):
                loaded = np.load(os.path.join(data_root, fi), allow_pickle = True)
                allData[loaded['label'][0]].extend(loaded['data'])
                allLabel[loaded['label'][0]].extend(loaded['label'])
        print('data from ./pre loaded')
        for i in range(10):
            allData[i] = sample(allData[i],15000)
            allLabel[i] = allLabel[i][:15000] #simple sampling are done here, will require more precise sampling
        allData = np.array(allData)
        allLabel = np.array(allLabel) #shape: (10, 15000, 100) (class, smaple, length)
        allData = allData.reshape(-1, 100)
        allLabel = allLabel.reshape(-1) #shape:(150000, 100)
        self.data = allData
        self.label = allLabel
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.label[index])
        #img = torch.from_numpy(img)
        #img = img.unsqueeze(0)
        #img = img.type(torch.FloatTensor)
        #img = Image.fromarray(img, mode='L') #for img
        #"""
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.type(torch.FloatTensor)
        #"""
        return img, target
    
    @property
    def train_labels(self):
        return self.label

    @property
    def test_labels(self):
        return self.label

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data
