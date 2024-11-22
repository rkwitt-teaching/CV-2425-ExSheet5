"""Submission for exercise sheet 5

SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_1234567.py
"""

import torch
import torch.nn as nn
from utils import get_data
from torch.utils.data import DataLoader


# Exercise 5.1
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()    
        #<YOUR CODE GOES HERE>

    def forward(self, x):
        #<YOUR CODE GOES HERE>
        pass
    
    
def train():
    ds_trn, _ = get_data()
    dl_trn = DataLoader(ds_trn, batch_size=32, shuffle=True)
    
    #<YOUR CODE GOES HERE>
    pass