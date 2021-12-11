import torch
import torch.nn as nn


class RNN_layer(nn.Module):
    
    def __init__(self, x_dim, h_dim):
        """
        Layer of a standard Recurrent Neural Network
        Expetced as outputs two torch.tensor of shapes  (num_examples, Tx, x_dim) and (num_examples, Tx, h_dim)
        where Tx is the length of the temporal sequenxe and num_examples is the mini_batch lenght
        Parameters:
        x_dim = dimension of feature vector
        h_dim = dimension of hidden state vector
        """
    
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

    
        



class LSTM_layer(nn.Module):

    def __init__(self, x_dim, h_dim):
        """
        Layer of a Long Short Term memory
        Expetced as outputs two torch.tensor of shapes  (num_examples, T, x_dim) and (num_examples, T, h_dim)
        where T is the length of the temporal sequence and num_examples is the mini_batch length
        Parameters:
        x_dim = dimension of feature vector
        h_dim = dimension of hidden state vector
        """
        
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
      
    def forward(self, x, a, c):
        """
        Forward pass of the LSTM
        Parameters:
        x = input torch.tensor of shape (num_examples, T, x_dim)
        a = hidden state torch.tensor of shape (num_examples, T. h_dim)
        c = cell state of shape (num_examples, T. h_dim)
        """
        
        # Input concatenation
        in_cat = torch.cat((a,x), dim=-1)
        
        # Forget gate
        f = self.sigmoid(self.forget_gate(in_cat))
        
        # Input gate
        i = self.sigmoid(self.input_gate(in_cat))
        
        # Cell update
        c = f*c + i*self.tanh(self.cell_update(in_cat))
      
        # Output gate
        o = self.sigmoid(self.out(in_cat))
        a = o*self.tanh(c)
        
        return o, a , c
      
      
      
 
