import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from copy import deepcopy
from torch import Tensor

class MLPNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(MLPNet, self).__init__()
        modules = []
        hidden_dims = deepcopy([512])
        hidden_dims.insert(0, input_dim)
        
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]
            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3))
            )
        self.net = nn.Sequential(*modules)
        self.output = nn.Linear(hidden_dims[-1], output_dim)
    def forward(self, x: Tensor):
        embedding = self.net(x)
        output = self.output(embedding)
        return output


class CNNNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512 * (input_dim // 2), 512) 
        self.fc2 = nn.Linear(512, output_dim)
    def forward(self, x: Tensor):
        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 


class TransformerNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, num_layers: int = 6):
        super(TransformerNet, self).__init__()   
        self.embedding = nn.Embedding(input_dim, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(512, output_dim)
    def forward(self, x: Tensor):
        x = self.embedding(x) 
        x = x.permute(1, 0, 2)       
        x = self.transformer_encoder(x)        
        x = x[-1, :, :]
        x = self.fc_out(x)
        return x