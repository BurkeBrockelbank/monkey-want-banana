"""
The monkey module contains the monkey object which holds a brain and keeps
track of food, age, etc.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/monkey.py
"""

from __future__ import print_function
from __future__ import division

import global_variables as gl
import exceptions

class Monkey:
    """
    This monkey runs on a deep Q neural nework to move around collecting bananas.
    """
    def __init__(self, brain):
        """
        Initialize the monkey's state.

        Args:
            brain: A brain object.
        """
        self.food = 20
        self.dead = False
        self.pos = (0,0)
        self.food_per_banana = 5
        self.food_per_turn = 1
        self.brain = brain
        # Epsilon initialized to -1 guarantees that the policy will never
        # choose a random move.
        self.bananas = 0
        self.age = 0

    def __repr__(self):
        return repr(self.brain)+str(self)

    def __str__(self):
        return 'food '+str(self.food)+\
        '\ndead '+str(self.dead)+\
        '\npos '+str(self.pos)+\
        '\nfood_per_banana '+str(self.food_per_banana)+\
        '\nfood_per_turn '+str(self.food_per_turn)+\
        '\nbananas '+str(self.bananas)+\
        '\nage '+str(self.age)

    def eat(self, n):
        """
        Run this function to give the monkey bananas.

        Args:
            n: The number of bananas to give the monkey.

        Returns:
            0: The number of bananas the monkey didn't want.
        """
        # Eat all of the bananas
        self.bananas += n
        self.food += self.food_per_banana*n
        return 0

    def action(self, s, epsilon = -1):
        """
        This passes the input vector to the Brain to evaluate the policy.

        Args:
            s: The vector that contains the information for vision.
            epsilon: Default -1. This is the chance of doing something random.

        Returns:
            0: The policy action in the form of an action integer.
        """
        if self.dead:
            return ' '
        # Get the policy
        a = self.brain.pi(s, epsilon)
        return a

    def tick(self):
        """
        This function consumes food and ages the monkey.
        """
        self.food -= self.food_per_turn
        self.age += 1
        if self.food < 0:
            self.die()

    def die(self):
        """
        Kill the monkey.
        """
        self.dead = True

    def move(self, action):
        """
        This function moves the monkey one space.

        Args:
            action: The action to move in terms of a 2-tensor where
                each row is one-hot.
        """
        symbol = gl.WASD[action]
        if self.dead:
            pass
        elif symbol  == 'a':
            self.pos = (self.pos[0], self.pos[1]-1)
        elif symbol  == 'd':
            self.pos = (self.pos[0], self.pos[1]+1)
        elif symbol  == ' ':
            self.pos = (self.pos[0], self.pos[1])
        elif symbol  == 'w':
            self.pos = (self.pos[0]-1, self.pos[1])
        elif symbol  == 's':
            self.pos = (self.pos[0]+1, self.pos[1])

    def unmove(self, action):
        """
        This function reverses the effect of the move function.

        Args:
            action: The action to unmove in terms of a one-hot vector.
        """
        symbol = gl.WASD[action]
        if self.dead:
            pass
        elif symbol  == 'd':
            self.pos = (self.pos[0], self.pos[1]-1)
        elif symbol  == 'a':
            self.pos = (self.pos[0], self.pos[1]+1)
        elif symbol  == ' ':
            self.pos = (self.pos[0], self.pos[1])
        elif symbol  == 's':
            self.pos = (self.pos[0]-1, self.pos[1])
        elif symbol  == 'w':
            self.pos = (self.pos[0]+1, self.pos[1])


