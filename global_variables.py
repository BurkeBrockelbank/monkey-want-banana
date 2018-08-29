"""
This file contains all global variables used for running the program.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/global_variables.py
"""

from __future__ import print_function
from __future__ import division

import torch
import random

# SIGHT =[[0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
#         [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
#         [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
#         [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
#         [0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0]]

# SIGHT =[[0,0,0,1,0,0,0],
#         [0,1,1,1,1,1,0],
#         [0,1,1,1,1,1,0],
#         [1,1,1,1,1,1,1],
#         [0,1,1,1,1,1,0],
#         [0,1,1,1,1,1,0],
#         [0,0,0,1,0,0,0]]

# SIGHT =[[0,0,0,1,1,1,1,1,0,0,0],
#         [0,0,1,1,1,1,1,1,1,0,0],
#         [0,1,1,1,1,1,1,1,1,1,0],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [0,1,1,1,1,1,1,1,1,1,0],
#         [0,0,1,1,1,1,1,1,1,0,0],
#         [0,0,0,1,1,1,1,1,0,0,0]]
SIGHT11 = torch.ones((11,11), dtype=torch.uint8)


SIGHT5 = torch.ones((5,5), dtype=torch.uint8)
# Note: SIGHT must be square and have an uneven number of rows.
SIGHT = SIGHT11
RADIUS = len(SIGHT)//2

# Define the block types
BLOCK_TYPES = ['#', 'm', 'b', 'd']
EMPTY_SYMBOL = ' '

# Get indeces for the block types
INDEX_BARRIER = BLOCK_TYPES.index('#')
INDEX_MONKEY = BLOCK_TYPES.index('m')
INDEX_BANANA = BLOCK_TYPES.index('b')
INDEX_DANGER = BLOCK_TYPES.index('d')

# Define movement symbols
WASD = 'wasd '

# Reward
DEATH_REWARD = -10

# Start with the basic room
ROOM_START_ASCII =  '##################################\n'+\
                    '#  b                   #  b      #\n'+\
                    '#          d  b        #     b   #\n'+\
                    '#    b                  d        #\n'+\
                    '#                 b    #         #\n'+\
                    '###########     b      #    d  b #\n'+\
                    '#          #           #  b      #\n'+\
                    '#  b        #      ########    ###\n'+\
                    '#    b           #      #        #\n'+\
                    '#          b    d     b        b #\n'+\
                    '#              #         b       #\n'+\
                    '#      d      #    dd      b     #\n'+\
                    '#    b d     #                   #\n'+\
                    '#      d          b      b     ###\n'+\
                    '#         b     #           ######\n'+\
                    '##################################'