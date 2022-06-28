import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(1, 64, 4, stride=4), nn.PReLU(),
                                     #nn.Dropout(0.05),
                                     nn.Conv1d(64, 128, 2, stride=2), nn.PReLU(),)
                                     #nn.MaxPool1d(4))

        self.fc = nn.Sequential(nn.Linear((128 * 12), 768), 
                                nn.PReLU(),
                                nn.Linear(768, 384),
                                nn.PReLU(),
                                nn.Linear(384, 192),
                                nn.PReLU(),
                                nn.Linear(192, 96),
                                nn.PReLU(),
                                nn.Linear(96, 10)
                                )
  
    
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

