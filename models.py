import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.metrics.functional import accuracy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import torch.nn.functional as F


class RNN(LightningModule):
    
    def __init__(self, x_dim, h_dim):
    
        super().__init__()
        
        # Parameters
        self.x_dim = x_dim
        self.h_dim = h_dim 
        
        # Input layers
        self.gate = nn.Linear(in_features = self.x_dim+self.h_dim, out_features = self.h_dim)
        
        # Output layer
        self.out = nn.Linear(in_features = self.h_dim, out_features = self.x_dim)
        
        # Act function
        self.act = nn.Tanh()
        
        
    def forward(self, x, h):
        
        # Concatenate input 
        in_cat = torch.cat((x,h), dim=-1)
        
        # Hidden and output
        h = self.act(self.gate(in_cat))
        o = self.out(h)
 
        return (o,h) 

    
        



class LSTM(LightningModule):

    def __init__(self, x_dim, h_dim):
        
        super().__init__()
        
        # Parameters
        self.x_dim = x_dim
        self.h_dim = h_dim 
        
        
        # Forget gate
        self.forget_gate = nn.Linear(in_features = self.x_dim+self.h_dim, out_features = self.h_dim)
        
        # Input gate
        self.input_gate = nn.Linear(in_features = self.x_dim+self.h_dim, out_features = self.h_dim)
        
        # Cell update
        self.cell_update = nn.Linear(in_features = self.x_dim+self.h_dim, out_features = self.h_dim)
        
        # Output gate
        self.out = nn.Linear(in_features = self.x_dim+self.h_dim, out_features = self.h_dim)
    
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
      
    def forward(self, x, h, C):
        
        # Input concatenation
        in_cat = torch.cat((x,h), dim=-1)
        
        # Forget gate
        f = self.sigmoid(self.forget_gate(in_cat))
        
        # Input gate
        i = self.sigmoid(self.input_gate(in_cat))
        
        # Cell update
        C = f*C + i*self.tanh(self.cell_update(in_cat))
      
        # Output gate
        o = self.sigmoid(self.out(in_cat))
        h = o*self.tanh(C)
        
        return o, h , C 
      
      
      
 
