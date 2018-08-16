"""
This file contains custom layers for use in monkey brains.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/layers.py
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class FoodWeight(nn.Module):
    """
        Applies a food-dependent weighting to a single channel.
    """
    def __init__(self, channels, height, width, bias = True):
        super(FoodWeight, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.food_weight = Parameter(torch.Tensor(1))
        self.channel_weights = Parameter(torch.Tensor(self.channels,1,1))
        self.bias = Parameter(torch.Tensor(1))

    def reset_parameters(self):
        self.food_weight.data.uniform_(-1, 1)
        self.food_weight.data.uniform_(-1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-1, 1)

    def forward(self, food, channel_map):
        """
        Forward pass through this layer.

        Args:
            food: Integer food level.
            channel_map: The torch.ByteTensor channel map.

        Returns:
            0: Weighted channel map of type torch.FloatTensor.
        """
        # Weight the food and channel
        channels_weighted = self.channel_weights.expand_as(channel_map) \
            * channel_map.type(torch.FloatTensor)
        food_weighted = food * self.food_weight
        # Add the weighted food and channel
        food_channel_sum = channels_weighted + \
            food_weighted.expand_as(channels_weighted)
        # Add in the bias
        return food_channel_sum + \
            self.bias.expand_as(food_channel_sum)

    def extra_repr(self):
        return 'height={}, width={}, food_weight={}, channel_weights={}, \
            bias={}'.format(self.height, self.width, self.food_weight, \
                self.channel_weights, self.bias is not None)




