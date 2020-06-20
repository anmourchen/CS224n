#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Definition of character level CNN architecture
    """
    def __init__(self, input_channels, output_channels):
        """
        Initialize the CNN architecture
        @params input_channels (int): number of channels in the input tensor
                output_channels (int): number of filters in the 1D convolution
        """
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=5)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x_reshaped):
        """
        Forward pass in the CNN arcitecture that maps x_reshaped into x_conv_out
        x_reshaped -> conv1d -> relu -> maxpool -> x_conv_out
        @param x_reshaped (torch.tensor): input tensor, shape: (src_len * batch_size, e_char, m_word)
        @returns x_conv_out (torch.tensor): output tensor, shape: (src_len * batch_size, e_word)
        """
        x_conv = F.relu(self.conv1d(x_reshaped))
        x_conv_out = self.maxpool(x_conv).squeeze(dim=2)
        return x_conv_out

### END YOUR CODE

