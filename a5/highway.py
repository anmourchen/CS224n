#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn.functional as F
import torch.nn as nn

class Highway(nn.Module):
    """
    Definition of Highway Networks
    """
    def __init__(self, input_size):
        """
        Initialize the highway networks
        @param input_size (int): Size of the feature dimensions, e_word in the pdf 
        """
        super(Highway, self).__init__()
        self.proj = nn.Linear(in_features=input_size, out_features=input_size)
        self.gate = nn.Linear(in_features=input_size, out_features=input_size)

    def forward(self, x_conv_out):
        """
        Forward pass of the highway block
        @param x_conv_out (torch.tensor): input tensor from convolution layers, shape: (input_size)
        @returns x_highway (torch.tensor): output tensor after passing the highway block, shape: (input_size)
        """
        x_proj = F.relu(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return x_highway

### END YOUR CODE 

