import torch
import torch.nn as nn
import numpy as np

class LogitScaleNetwork(nn.Module):
    def __init__(self, init_scale=1 / 0.07):
        super(LogitScaleNetwork, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(init_scale)) # from openclip
        
    def forward(self, x=None): # add x to make it compatible with DDP
        return self.logit_scale.exp()