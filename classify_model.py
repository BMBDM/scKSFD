import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from copy import deepcopy
from torch import Tensor

class Net(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Net, self).__init__()
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
    