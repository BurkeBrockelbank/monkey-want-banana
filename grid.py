"""
This is the grid module, which contains the grid class. Grid classes manage the
game state.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/grid.py
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import global_variables as gl
import exceptions
import room_generator as rg

import math
import random
import copy

class Grid:
    """
    The grid object keeps track of and alters the game state. All queries
    actions related to the game state should go through here.

    Eventually this object will support multiple monkeys. Efforts are made to
    enforce this capability but as of now, this object is only tested for a
    single monkey.
    """
    def __init__(self, monkeys, channel_map):
        """
        Initialization for the grid object.

        Args:
            monkeys: A list of monkeys.
            room: A channel map. Data type is unit8.
        """
        self.monkeys = monkeys
        self.channel_map = channel_map
        self.height = len(channel_map[1])
        self.width = len(channel_map[2])
        self.turn_count = 0
        # Add in the monkeys into the room
        self.replace_monkeys()

    def replace_monkeys(self):
        """
        Puts the monkeys in their place in the channel map.\
        """
        # Clear all the monkeys
        self.channel_map[gl.INDEX_MONKEY] = torch.zeros(\
            self.channel_map[0].size(), dtype = torch.uint8)
        for monkey in self.monkeys:
            i,j = monkey.pos
            self.channel_map[gl.INDEX_MONKEY,i,j] += 1

    def teleport_monkey(self, ij0, ij1):
        """
        Used to update the position of a single monkey in the channel map.

        Args:
            ij0: The old position of the monkey (2 tuple of integers)
            ij1: The new position of the monkey (2 tuple of integers)

        Raises:
            IndexError: If there wasn't a monkey there to begin with.
        """
        if self.channel_map[gl.INDEX_MONKEY,ij0[0],ij0[1]] > 0:
            self.channel_map[gl.INDEX_MONKEY,ij0[0],ij0[1]] -= 1
            self.channel_map[gl.INDEX_MONKEY,ij1[0],ij1[1]] += 1
        else:
            raise IndexError('No monkey to remove at' + str(ij0))

    def tick(self, control, directions = [], invincible = False, loud=[], wait=True):
        """
        This function moves the entire grid and all the monkeys forward one
        timestep.

        Args:
            control: If 0, the monkey is queried on its moves. If 1,
                a list of movement directions must be given which is the same
                length as self.monkeys. If 2, the user is queried for a
                movment direction.
            directions: A list of actions. One for each monkey. Only applies if
                control is 1.
            invincible: Default False. If true, the monkey is not removed if it
                dies.
            loud: Default []. This list is a list of the indeces of monkeys to watch.
            wait: Default True. If false, doesn't wait for user when control is
                0 or 1.

        Raises:
            ControlError: Raised if control is not properly defined.
        """
        # Report the turn if set to loud
        if loud != []:
            print('TURN', self.turn_count)

        # Instantiate a list for dying monkeys
        dead_monkeys = []

        # Instantiate record lists
        foods = []
        actions = []
        surrs = []

        # Iterate through all the monkeys
        for monkey_index, monkey in enumerate(self.monkeys):
            # Get the surroundings of the monkey.
            surr = self.surroundings(monkey.pos)
            # Print details for this tick
            if monkey_index in loud:
                # Print monkey number and number of bananas
                print('Monkey', monkey_index, 'food', int(monkey.food), 'age', monkey.age)
                # Get the ascii map
                text_map = rg.channel_to_ASCII(surr,indeces=True,index_offset=monkey.pos)
                # Print the ascii map
                print(text_map)

            # Determine control type and get action
            if control == 2:
                # Get user input.
                # Loop until good input is given
                need_input = True
                while need_input:
                    # Get action
                    action_string = input('>>>')
                    # Turn action string into an action integer
                    try:
                        action = gl.WASD.index(action_string)
                        need_input = False
                    except ValueError:
                        print('Input must be w, a, s, d, or space.')
            elif control == 1:
                # Get input from a list of directions
                try:
                    action = directions[monkey_index]
                    # Get the string action
                    action_string = gl.WASD[action]
                except IndexError as e:
                    raise ControlError('Directions not specified').\
                        with_traceback(e.__traceback__)
                if action not in range(len(gl.WASD)):
                    raise ControlError('Action ' + str(action) + \
                        ' for monkey ' + str(monkey_index)+' is not valid.')
                # Print out the action if loud and wait are is on
                if monkey_index in loud:
                    if wait:
                        input('>>>'+action_string)
                    else:
                        print('>>>'+action_string)
            elif control == 0:
                # Get action from monkey's brain
                Q, action, probability = monkey.brain.pi((monkey.food, surr))
                # Get the string action
                action_string = gl.WASD[action]
                # Print out the action if loud and wait are is on
                if monkey_index in loud:
                    if wait:
                        input('>>>'+action_string+' '+str(probability))
                    else:
                        print('>>>'+action_string+' '+str(probability))
            else:
                raise ControlError('Control must be specified as 0, 1, or 2')

            # Add the surroundings and actions to the record.
            foods.append(monkey.food)
            actions.append(action)
            surrs.append(surr)

            # Now we want to move the monkey
            monkey.move(action)
            # Get the blocks on this space
            this_space = self.channel_map[:,monkey.pos[0],monkey.pos[1]]
            # Check if the monkey is trying to move to a barrier
            if this_space[gl.INDEX_BARRIER] >= 1:
                # Need to unmove the monkey.
                monkey.unmove(action)
                # Get the blocks on this space
                this_space = self.channel_map[:,monkey.pos[0],monkey.pos[1]]

            # Feed the monkey any bananas on this spot
            remaining_bananas = \
                monkey.eat(int(this_space[gl.INDEX_BANANA]))
            eaten_bananas = this_space[gl.INDEX_BANANA] - \
                remaining_bananas
            # Randomly put new bananas around
            for banana_index in range(eaten_bananas):
                banana_placed = False
                while not banana_placed:
                    # Get two random indeces
                    i = random.randrange(self.height)
                    j = random.randrange(self.width)
                    # Make sure the spot is empty
                    empty_spots = self.channel_map[:,i,j] == \
                        torch.zeros(len(gl.BLOCK_TYPES), dtype = torch.uint8)
                    # Bananas and monkeys are alright, but we can't put a
                    # banana in a barrier or danger.
                    barrier_empty = empty_spots[gl.INDEX_BARRIER]\
                        .item()
                    danger_empty = empty_spots[gl.INDEX_DANGER]\
                        .item()
                    banana_placed = bool(barrier_empty and danger_empty)
                # Put a banana there
                self.channel_map[gl.INDEX_BANANA,i,j] += 1
            # Remove all the eaten bananas on this spot
            this_space[gl.INDEX_BANANA] = remaining_bananas

            # Check if the monkey is in danger
            if this_space[gl.INDEX_DANGER] >= 1:
                monkey.dead = True
            # The monkey now ages and consumes food (possibly starving to death).
            monkey.tick()
            # Clean up the monkey if need be
            if monkey.dead:
                # Mark monkey for cleanup
                dead_monkeys.append(monkey_index)
        # Remove dead monkeys
        for dead_index in dead_monkeys[::-1]:
            if not invincible:
                del self.monkeys[dead_index]
        # Replace the monkeys
        self.replace_monkeys()
        # Check if there are any monkeys left
        if len(self.monkeys) == 0:
            print('All monkeys have died.')

        # Update the turn count.
        self.turn_count += 1
        return foods, actions, surrs


    def surroundings(self, pos):
        """
        This function finds the surroundings of a monkey based on its
        sightlines. The sightline is assumed to be a square matrix.

        Args:
            pos: The position (integer couple) around wich we will center the
                map.

        Returns:
            0: A cropped channel map that has been obscured according to
                gl.SIGHT.
        """
        # The first thing to do is pad the channel map with enough zeros that
        # we could put the monkey anywhere and still slice the array
        radius = len(gl.SIGHT)//2
        padded_size = torch.tensor(self.channel_map.size())
        padded_size += torch.tensor([0, radius*2, radius*2])
        padded_size = torch.Size(padded_size.tolist())
        padded = torch.zeros(padded_size,dtype=torch.uint8)
        for i in range(len(gl.BLOCK_TYPES)):
            # Pad the barrier channel with ones and everything else with zeros.
            padding_value = 0
            if i == gl.INDEX_BARRIER:
                padding_value = 1
            padded[i] = F.pad(self.channel_map[i],\
            (radius, radius, radius, radius), value=padding_value)
        # Note: The elements of padded do not share pointers with the elements
        # of self.channel_map.
        # Slice the array
        sliced = padded[:, pos[0]:pos[0]+len(gl.SIGHT), \
            pos[1]:pos[1]+len(gl.SIGHT)]
        # Now we need to obscure any blocks that are deemed invisible.
        for i, row in enumerate(gl.SIGHT):
            for j, el in enumerate(row):
                # An invisible block is marked as a zero in SIGHT
                if el == 0:
                    # Turn this spot into a barrier
                    for k in range(len(gl.sight)):
                        if k == gl.index('#'):
                            sliced[k][i][j] = 1
                        else:
                            sliced[k][i][j] = 0
        # Return the surroundings
        return sliced


    def __str__(self):
        """
        This function returns the ASCII map.

        Returns:
            0: ASCII string.
        """
        return rg.channel_to_ASCII(self.channel_map)

    def __repr__(self):
        """
        This function returns the channel maps in a string plus the
        list of monkeys in strings.

        Returns:
            0: Representation string.
        """
        return repr(self.turn_count) + repr(self.channel_map) + \
            str(self.monkeys)





