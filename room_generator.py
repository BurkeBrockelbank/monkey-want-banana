"""
This is the room generator module. This module contains function for converting
between ASCII art maps and the maps that are used for the program (called
channel maps).

IMPORTANT NOTE: Converting a channel map into an ASCII map removes information
about multiple objects in the same grid space. Converting an ASCII map into a
channel map loses no information.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/room_generator.py
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

import random
import numpy as np
from PIL import Image

import global_variables as gl
import exceptions

# Make the abstract and concretize functions
def ASCII_to_channel(ASCII_map):
    """
    Converts an ASCII map to a channel map.

    Args:
        ASCII_map: The ASCII map to convert.

    Returns:
        0: The channel map.

    Raises:
        MapSizeError: Raised if the ASCII map is not rectangular.
        SymbolError: If there are unrecognized symbols in the ASCII map.
    """
    # Split the ASCII map along newline characters.
    rows = ASCII_map.split('\n')
    # Get the height and width
    height = len(rows)
    width = len(rows[0])

    # Make sure the map is rectangular
    for row in rows:
        try:
            assert len(row) == width
        except AssertionError as e:
            raise exceptions.MapSizeError('ASCII map  is not rectangular.').\
                with_traceback(e.__traceback__)

    # Initialize the channel map
    channels = torch.zeros((len(gl.BLOCK_TYPES),height, width), dtype = torch.uint8)
    # Iterate through the ASCII map
    for i, row in enumerate(rows):
        for j, symbol in enumerate(row):
            # Find the appropriate channel corresponding to this block type
            # Skip this for the empty symbol.
            if symbol != gl.EMPTY_SYMBOL:
                try:
                    symbol_index = gl.BLOCK_TYPES.index(symbol)
                except ValueError as e:
                    raise exceptions.SymbolError('Symbol '+str(symbol)+\
                        ' is not recognized.').with_traceback(e.__traceback__)
                # Mark the channel with a 1 at the position of the block
                channels[symbol_index][i][j] += 1

    return channels

def channel_to_ASCII(channel_map,indeces=False,index_offset=(0,0)):
    """
    This funciton converts a channel map to an ASCII map representation.

    Args:
        channel_map: The channel map in question.
        indeces: Default False. If True, show indeces in the map.
        index_offset: Default (0,0). If indeces is True, then this offests the
            indeces in the output ASCII map.

    Returns:
        0: ASCII map string.

    Raises:
        MapSizeError: Raised if the channel map has inconsistent sizing.
    """
    # If we are asking for indeces, we will first need to call this function
    # with no optional arguments to get the basic map.
    if indeces:
        # Calculate the size of the sightrange
        radius = len(gl.SIGHT)
        # Get the basic map through a recursive call and split into rows
        basic_map = channel_to_ASCII(channel_map).split('\n')
        # Initialize the map picture
        map_picture = ''
        # The first row shows the digits in the tens place and higher
        d10 = lambda x : x-x%10
        map_picture += ' '*len(str(d10(index_offset[0])))+'  '+\
            str(d10(index_offset[1]))+'+\n '+\
            ' '*len(str(d10(index_offset[0])))+' '
        # The second row has the ones digit
        for j in range(len(basic_map[0])):
            map_picture += str((j+index_offset[1]-radius)%10)
        map_picture += '\n'
        # Now we need to find the tens and higher places for the vertical
        # direction.
        for i in range(len(basic_map)):
            if i==0:
                map_picture += str(d10(index_offset[0]))+'+'
            else:
                map_picture += ' '*len(str(d10(index_offset[0])))+' '
            map_picture += str((i+index_offset[0]-radius)%10)
            map_picture += basic_map[i]
            map_picture += '\n'
        return map_picture.rstrip()

    # Otherwise, we need to make the basic map
    else:
        # Find the size of the channel map
        height = len(channel_map[0])
        width = len(channel_map[0][0])

        # Assert that the shape is correct
        try:
            assert len(channel_map) == len(gl.BLOCK_TYPES)
        except AssertionError as e:
            raise exceptions.MapSizeError('Channel map has extra channels.').\
                with_traceback(e.__traceback__)
        try:
            for channel in channel_map:
                    assert len(channel) == height
                    for row in channel:
                        assert len(row) == width
        except AssertionError as e:
            raise exceptions.MapSizeError('Channel map is not rectangular.').\
                with_traceback(e.__traceback__)

        # Create a blank map
        ASCII_rows = [[gl.EMPTY_SYMBOL]*width for i in range(height)]
        # Iterate through the blank map
        for i, row in enumerate(ASCII_rows):
            for j in range(len(row)):
                # Get a list of all the blocks in this spot.
                channels = [channel[i][j] for channel in channel_map]
                # Get the index of the maximal element.
                # This corresponds to the type of block which is most
                # populous in this position.
                maxIndex = max(range(len(gl.BLOCK_TYPES)), \
                    key = channels.__getitem__)
                # If there is no block here, do nothing
                if channels[maxIndex] != 0:
                    # If there is any danger here, it needs to be shown,
                    # regardless of the maximum index.
                    if channels[gl.INDEX_DANGER] > 0:
                        symbol = 'd'
                    else:
                        # Otherwise we will add the block here
                        symbol = gl.BLOCK_TYPES[maxIndex]
                    # Add the block to the map
                    ASCII_rows[i][j] = symbol
        # Join the rows together
        for i, row in enumerate(ASCII_rows):
            ASCII_rows[i] = ''.join(row)
        # The map has been populated now. Just add the newline characters,
        # concatenate it together, and return it.
        return '\n'.join(ASCII_rows)


def rand_room(size, rates):
    # Avoid infinite loop
    try:
        assert sum(rates) < 1
    except AssertionError:
        raise exceptions.MapSizeError('Rates must sum to less than one' +\
            'to fit all blocks in map.')

    # Build empty room
    room = torch.zeros((len(gl.BLOCK_TYPES), size, size), \
    dtype=torch.uint8)

    # Build barriers on edges
    for i in [0,-1]:
        for j in range(size):
            room[gl.INDEX_BARRIER, i, j] = 1
            room[gl.INDEX_BARRIER, j, i] = 1

    # Populate room
    n_blocks = (size-1)**2
    for block_index, block_type in enumerate(gl.BLOCK_TYPES):
        rate = rates[block_index]
        for n in range(round(n_blocks*rate)):
            empty_spot = False
            while not empty_spot:
                i = random.randrange(size)
                j = random.randrange(size)
                empty_spot = all(room[:,i,j] == 0)
            room[block_index,i,j] = 1

    return room

def play_record(path):
    """
    This function reads in the training data from a path and replays it for
    viewing.

    Args:
        path: The path to the data.
    """
    in_f = open(path, 'r')
    in_lines = in_f.readlines()
    in_f.close()
    # parse the input lines
    data = [eval(x.rstrip()) for x in in_lines]
    # As a reminder, the data structure is
    # food (int), action (int), Qualities, board state (torch.tensor dtype=torch.uint8)

    # Iterate through the data
    for x in data:
        food = x[0]
        action = x[1]
        board = x[2]
        print('food', food)
        print(channel_to_ASCII(board))
        input('>>>' + gl.WASD[action])

def png_to_channel(path, rgb_list):
    """
    This function reads a png and turns it into a channel map.

    Args:
        path: The path to the png image.
        rgb_list: A parallel list with gl.BLOCK_TYPES that says what the rgb values
            of each block type are. List of tuples.

    Returns:
        0: Channel map.
    """
    # Turn rgb list into tensor
    rgb = torch.tensor(rgb_list, dtype=torch.uint8)
    # Open the file as a tensor
    img = Image.open(path)
    img.load()
    img = img.convert('RGB')
    width, height = img.size
    img = list(img.getdata())
    img = torch.tensor(img, dtype=torch.uint8)
    img = img.view((height,width,3))
    # Create a new tensor to be the channel map
    channels = torch.zeros((len(gl.BLOCK_TYPES), height, width), dtype=torch.uint8)
    # Populate the channel map
    for layer in range(len(gl.BLOCK_TYPES)):
        for i in range(height):
            for j in range(width):
                if all(img[i,j] == rgb[layer]):
                    channels[layer, i, j] = 1
    return channels

def free_spot(channel_map):
    while 1:
        i = np.rand(channel_map.size()[1])
        j = np.rand(channel_map.size()[2])
        if all(channel_map[:,i,j] == 0):
            return i,j