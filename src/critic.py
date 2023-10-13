import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


'''
all critics have the same network architecture, 
which is composed of one input layer, 
multiple hidden layers with Leaky ReLU activation functions, 
and one output layer with linear activation function
'''

class critic(nn.Module):
    def __init__(self,n_observations ) -> None:
        super(critic,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations,128),
            nn.LeakyReLU(),
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128,1),
        )
        self.lr = 0.001
    
    def forward(self,x):
        return self.net(x)