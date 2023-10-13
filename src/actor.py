'''
Note that all actors have the same network architecture, 
which consists of one input layer,
multiple hidden layers with Leaky ReLU activation functions, 
and one output layer with softmax activation function
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class actor(nn.Module):
    def __init__(self,n_observations, n_actions) -> None:
        super(actor,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations,128),
            nn.LeakyReLU(),
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128,n_actions),
            nn.Softmax(dim=1)
        )
        self.lr = 0.0005
    
    def forward(self, x):
        return self.net(x)