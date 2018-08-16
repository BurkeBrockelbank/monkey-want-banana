"""
This module holds brain type objects. Brain objects must have a forward
function that takes in a tensor of states and returns a one-hot direction
vector. They must have a policy function that evaluates the highest
quality move. They must have a Q function that takes in a tensor of states
and one-hot direction and returns the quality of that move.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/brain.py
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import global_variables as gl
import exceptions
import room_generator as rg

import random
import bisect
import itertools

class BrainDQN(nn.Module):
    """
    This is the basic deep-Q brain class that all other brain classes will be
    based on.

    This brain has no memory implementation.

    The convention will be that every Q, s, a or other expression relating to a
    single state of phase space will have no dimension for multiples, i.e. they
    will NOT be in the form tensor([s1, s2, s3, s4, ...]) where sn is the nth
    state.

    s is a tuple (food, vision).

    food is an integer.

    vision is a 3-tensor. The first index refers to channel, the second to row, and
    the third to column.

    a is an integer 1-tensor corresponding to the
    index of the direction as specified in gl.WASD.

    Qs is a 1-tensor of length five.

    Q is a 0-tensor float corresponding to the quality of an action in a
    given state.
    """

    def __init__(self):
        """
        Initialize the architecture of the neural net.
        """
        # Initialize the parent class
        super(BrainDQN, self).__init__()

    def forward(self, s):
        """
        Returns the 5-tensor of qualities corresponding to each direction.

        Args:
            s: The state of the system.
        
        Returns:
            0: 5-tensor of qualities.

        Raises:
            BrainError: This is only meant to be a parent class. Using this class
            as if it were functional should raise an error.
        """
        raise exceptions.BrainError

    def pi_greedy(self, s):
        """
        Gives the quality and action corresponding to a totally greedy policy.

        Args:
            s: The state of the system.

        Returns:
            0: The maximal quality.
            1: The action that maximizes quality.
        """
        Qs = self.forward(s)
        max_Q, max_a = Qs.max(0)
        return max_Q, max_a, 1

    def argmax_a(self, s):
        """
        Returns the action that maximizes quality for the state s.
        
        Args:
            s: The state of the system.

        Returns:
            0: Action which maximizes the quality of the state.
        """
        max_Q, max_a, _ = self.pi_greedy(s)
        return max_a

    def Q(self, s, a):
        """
        Returns the quality of performing action a in state s.

        Args:
            s: The state of the system.
            a: The action.

        Returns:
            0: The quality of that move.
        """
        Qs = self.forward(s)
        return Qs[a]


    def pi_epsilon_greedy(self, s, epsilon):
        """
        Enacts the epsilon-greedy policy of the brain.

        Args:
            s: The state of the system.
            epsilon: The probibility of taking a random movement.

        Returns:
            0: The determined quality of this move.
            1: The action the policy points to.
        """
        # Roll a random number
        if random.uniform(0, 1) < epsilon:
            # Make a random movement
            Q, a = self.pi_random(s)
            # Calculate the qualities
            return Q, a, epsilon
        else:
            # Get the probability of this occuring
            p = 1-epsilon
            # Just take the best action
            Q, a, _ = self.pi_greedy(s)
            return Q, a, p

    def pi_probabilistic(self, s):
        """
        Enacts the policy of qualities corresponding to probabilities.
        Qualities are fed into a softmax and then an action selected
        according to those probabilities.

        Args:
            s: The gamesate.

        Returns:
            0: The quality of the action.
            1: The action to be taken.
        """
        # Find the qualities.
        Qs = self.forward(s)
        # Run this through a softmax
        Q_softmax = F.softmax(Qs, dim=0)
        # Calculate the CDF.
        CDF = [Q_softmax[0]]
        # The CDF is one element shorter than the probabilities because the
        # last element is one.
        for p in Q_softmax[1:-1]:
            CDF.append(CDF[-1]+p)
        # Generate a random number in a uniform distribution from 0 to 1.
        roll = random.uniform(0, 1)
        # Get the action this corresponds to.
        a = bisect.bisect(CDF, roll)
        # Return the quality and action.
        return Qs[a], a, Q_softmax[a]

    def pi_random(self, s):
        """
        Enacts the policy of qualities corresponding to random decision.

        Args:
            s: The gamesate.

        Returns:
            0: The quality of the action.
            1: The action to be taken.
        """
        # Make a random movement
        a = random.randrange(0,len(gl.WASD))
        # Calculate the qualities
        return self.Q(s,a), a


class BrainLinearAI(BrainDQN):
    """
    This is an antificial intelligence for moving towards bananas.

    No memory implementation.
    """

    def __init__(self):
        """
        There is no neural net to initialize, but we do need to initialize the
        parent class to inherit its user-defined functions.
        """
        BrainDQN.__init__(self)

        # Calculate values for AI.
        self.value_factor = torch.zeros(len(gl.SIGHT),\
                                        len(gl.SIGHT),\
                                        len(gl.BLOCK_TYPES),\
                                        2)

        # Define the base values and powers
        T = torch.zeros((len(gl.BLOCK_TYPES),2,2))
        T[gl.INDEX_BARRIER] = torch.FloatTensor([[-2,0.1],
                                                [-0.1,-2]])
        T[gl.INDEX_MONKEY] = 0.5*torch.eye(2)
        T[gl.INDEX_BANANA] = torch.FloatTensor([[4.1, 0.1],
                                                 [0,4]])
        T[gl.INDEX_DANGER] = torch.FloatTensor([[-2.1, 1],
                                                 [-1,-2.1]])
        p = torch.zeros(len(gl.BLOCK_TYPES))
        p[gl.INDEX_BARRIER] = 2
        p[gl.INDEX_MONKEY] = 1
        p[gl.INDEX_BANANA] = 3
        p[gl.INDEX_DANGER] = 4

        # Calculate radius
        radius = len(gl.SIGHT)//2

        # Iterate through positions first.
        for i in range(len(gl.SIGHT)):
            for j in range(len(gl.SIGHT)):
                if i == radius and j == radius:
                    continue
                # The spot is populated. Calculate the vector.
                v = torch.FloatTensor([i-radius,j-radius])
                # Calculate the norm
                norm = v.norm()
                # Multiply the transformation
                Tv = torch.matmul(T,v.view(2,1)).view(len(gl.BLOCK_TYPES),2)
                # Divide by the norm appropriately
                for cc in range(len(gl.BLOCK_TYPES)):
                    Tv[cc] /= norm**p[cc]
                # Add to value factors
                self.value_factor[i,j,:,:] = Tv

    def pi(self, s):# Set default policy.
        return self.pi_greedy(s)

    def forward(self, s):
        """
        This is the value that the AI determines for each direction. It is
        analogous to a quality vector, except there is no interpretation of
        the value in the vector being related to the number of bananas
        we expect.

        Every block type is assigned a matrix T and decay p.

        Every object in the map contributes to the value by its own vector v
        measured with respect to the monkey.

        Value Contribution = T(v) / ||v||**p

        T_barrier = [[0,0.5],
                         [-0.5,0]]
        p_barrier = 4

        T_monkey = [[0.1,0],
                    [0,0.1]]
        p_monkey = 1

        T_banana = [[1,0],
                    [0,1]]
        p_banana = 3

        T_danger = -20*[[1,0],
                        [0,1]]
        p_danger = 4

        Monkeys have no base value.

        Bananas are viewed with a base value of 1 and are divided by magnitude
        cubed.

        Danger is viewed with a base value of -20 and divided by magnitude to
        the fourth power.

        The value of W, A, S, or D movements is determined this way. The value
        of staying still is zero, te preserve linearity.

        Args:
            s: The state of the system.
        
        Returns:
            0: 5-tensor of values.
        """

        # First get the channels map
        food, vision = s

        # Start of the value contribution sum as zero
        value_sum = torch.zeros(2)

        # Iterate through positions first.
        for i in range(len(gl.SIGHT)):
            for j in range(len(gl.SIGHT)):
                # Check if this spot is populated
                if 1 in (vision[:,i,j] > 0):
                    # The spot is populated. Calculate the vector.
                    value_add = torch.matmul(vision[:,i,j].float()[None],\
                        self.value_factor[i,j,:,:]).view(2)
                    value_sum += value_add
                    # if i in [3,4,5,6,7] and j in [3,4,5,6,7]:
                    #     print(i,j,value_add.tolist())


        # Now we just re-index the value sum
        values = torch.zeros(len(gl.WASD))
        values[0] = -value_sum[0]
        values[1] = -value_sum[1]
        values[2] = value_sum[0]
        values[3] = value_sum[1]

        # Return the values
        return values




class BrainLinear(BrainDQN):
    """
    This implements the naive approach of the monkey brain as described in the
    presentation. It is a simple linear model.

    This brain has no memory implementation.
    """

    def __init__(self):
        """
        Initialize the architecture of the neural net.
        """
        # Initialize the parent class
        BrainDQN.__init__(self)
        # Set the default policy
        self.pi = self.pi_epsilon_greedy

        # Initialize the neural network
        # We just need to take the input size:
        # 1 [food] + 11*11 [grid size] * 4 [number of channels] = 485
        self.layer = nn.Linear(485,5)

    def forward(self, s):
        """
        Returns the 5-tensor of qualities corresponding to each direction.

        Args:
            s: The state of the system.
        
        Returns:
            0: 5-tensor of qualities.
        """
        # Unpack the state
        food, vision = s
        
        # First we need to flatten out the state and add in the food.
        vision_float = vision.float().view(-1)
        food_float = torch.FloatTensor([food])
        state = torch.cat((food_float,vision_float),0)

        # Secondly we need to run this though the linear layer.
        Qs = self.layer(state)

        return Qs


class BrainV1(BrainDQN):
    """
    This implements the first approach of the monkey brain as described in the
    presentation.

    This brain has no memory implementation.
    """

    def __init__(self):
        """
        Initialize the architecture of the neural net.
        """
        # Initialize the parent class
        BrainDQN.__init__(self)
        # Set the default policy
        self.pi = self.pi_epsilon_greedy

        # Initialize the neural network
        # Part 1: Desire Mapping
        self.DM_weight = layers.FoodWeight(3, len(gl.SIGHT), len(gl.SIGHT[0]))
        self.DM_conv = nn.Conv2d(3, 1, 3, stride=2)
        # Part 2: Path Finding
        self.PF_conv1 = nn.Conv2d(2, 4, 3)
        self.PF_conv2 = nn.Conv2d(4, 1, 5)
        # Part 3: Evaluation
        self.EV_linear1 = nn.Linear(50,8)
        self.EV_linear2 = nn.Linear(8,5)

    def forward(self, s):
        """
        Returns the 5-tensor of qualities corresponding to each direction.

        Args:
            s: The state of the system.
        
        Returns:
            0: 5-tensor of qualities.
        """
        # Unpack the state
        food, vision = s
        
        # Part 1: Desire Mapping
        # Get the correct input channels.
        DM_channels = vision.index_select(0,torch.tensor(\
            [gl.INDEX_MONKEY, gl.INDEX_BANANA, gl.INDEX_DANGER]))
        # Weight the channels
        DM_weighted = self.DM_weight(food, DM_channels)
        DM_weighted = F.relu(DM_weighted)
        # Build desire map
        # Need to add another dimension with None indexing
        DM_desire_map = self.DM_conv(DM_weighted[None])

        # Part 2: Path Finding
        # Get the correct input channels
        PF_channels = vision.index_select(0,torch.tensor(\
            [gl.INDEX_BARRIER, gl.INDEX_DANGER]))
        # Find walls
        PF_walls = self.PF_conv1(PF_channels[None].type(torch.FloatTensor))
        PF_walls = F.sigmoid(PF_walls)
        # Determine passability
        PF_path_map = self.PF_conv2(PF_walls)
        PF_path_map = F.relu(PF_path_map)

        # Part 3： Evaluation
        # Flatten
        EV_flat = torch.cat((DM_desire_map.view(-1), PF_path_map.view(-1)), 0)
        # Fully connected ReLU layers
        h = F.relu(self.EV_linear1(EV_flat))
        Qs = F.relu(self.EV_linear2(h))

        return Qs

class BrainV2(BrainDQN):
    """
    This implements the second approach of the monkey brain. It is a more
    conventional CNN.

    This brain has no memory implementation.
    """
    def __init__(self, gamma):
        """
        Initialize the architecture of the neural net.
        """
        # Initialize the parent class
        BrainDQN.__init__(self)
        # Set the default policy
        self.pi = self.pi_epsilon_greedy

        # Create convolutional layers of neural network.
        self.l1 = nn.Conv2d(4,4,3,padding=1)
        self.l2 = nn.Conv2d(4,6,3)
        self.l3 = nn.Conv2d(6,4,3)
        self.l4 = nn.Conv2d(4,1,3)

        # Create fully connected layers
        self.l5 = nn.Linear(26,9)
        self.l6 = nn.Linear(9,8)
        self.l7 = nn.Linear(8,5)

        # Set final weights to reflect zero reward
        reward = 1/(gamma-1)
        nn.init.constant_(self.l7.bias, reward)

    def forward(self, s):
        """
        Args:
            s: The state of the system.
        
        Returns:
            0: 5-tensor of qualities.
        """
        # Unpack state
        food, vision = s

        # Convolutional layers
        h = F.relu(self.l1(vision[None].float()))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))

        # Fully connected layers
        # First flatten
        h = h.view(-1)
        # Add in food
        h = torch.cat((torch.FloatTensor([food]),h))
        h = F.relu(self.l5(h))
        h = F.relu(self.l6(h))
        Q = self.l7(h)

        return Q

class BrainProgression(BrainDQN):
    """
    This implements the second approach of the monkey brain. It is a more
    conventional CNN.

    This brain has no memory implementation.
    """
    def __init__(self):
        """
        Initialize the architecture of the neural net.
        """
        # Initialize the parent class
        BrainDQN.__init__(self)
        # Set the default policy
        self.pi = self.pi_epsilon_greedy

        # Create convolutional layers of neural network.
        self.c1 = nn.Conv2d(4,6,3,padding=1)
        self.c2 = nn.Conv2d(6,4,3,padding=1)
        self.c3 = nn.Conv2d(4,1,5)

        # Create fully connected layers
        self.f1 = nn.Linear(50,9)
        self.f2 = nn.Linear(9,8)
        self.f3 = nn.Linear(8,5)

    def forward(self, s):
        """
        Args:
            s: The state of the system.
        
        Returns:
            0: 5-tensor of qualities.
        """
        # Unpack state
        food, vision = s

        # Convolutional layers
        h = F.relu(self.c1(vision[None].float()))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))

        # Fully connected layers
        # First flatten
        h = h.view(-1)
        # Add in food
        h = torch.cat((torch.FloatTensor([food]),h))
        h = F.relu(self.f1(h))
        h = F.relu(self.f2(h))
        Q = self.f3(h)

        return Q

class BrainV3(BrainDQN):
    """
    This implements the third approach of the monkey brain. This allows fine
    features to be detected close to the monkey.

    This brain has no memory implementation.
    """
    def __init__(self):
        """
        Initialize the architecture of the neural net.
        """
        # Initialize the parent class
        BrainDQN.__init__(self)
        # Set the default policy
        self.pi = self.pi_epsilon_greedy

        # Fine branch

        # Convolutional branch
        self.c1 = nn.Conv2d(4,8,3,padding=1)
        self.c2 = nn.Conv2d(8,4,4)
        self.c3 = nn.Conv2d(4,2,4)

        # Create fully connected layers
        self.f1 = nn.Linear(151,25)
        self.f2 = nn.Linear(25,9)
        self.f3 = nn.Linear(9,8)
        self.f4 = nn.Linear(8,5)

        # report flag
        self.report = False

    def forward(self, s):
        """
        Args:
            s: The state of the system.
        
        Returns:
            0: 5-tensor of qualities.
        """
        # Unpack state
        food, vision = s
        food_float = torch.FloatTensor([food])
        vision_float = vision.float()[None]

        # Fine branch
        h_fine = vision_float[:,:,3:-3,3:-3]

        # Convolutional layers
        h_conv = F.relu(self.c1(vision_float))
        h_conv = F.relu(self.c2(h_conv))
        h_conv = F.relu(self.c3(h_conv))

        # Combine branches
        h = torch.cat((h_fine[0], h_conv[0]), dim=0)

        # Fully connected layers
        # First flatten
        h = h.view(-1)
        # Add in food
        h = torch.cat((torch.FloatTensor([food]),h))
        h = F.relu(self.f1(h))
        h = F.relu(self.f2(h))
        h = F.relu(self.f3(h))
        Q = self.f4(h)

        if self.report:
            print(Q)

        return Q

class BrainDecisionAI(BrainDQN):
    """
    This is an antificial intelligence for moving towards bananas. It moves
    towards the closest bananas and outputs a quality each time it moves.

    No memory implementation.
    """

    def __init__(self, gamma, r_b, r_e, r_d, save_Q=False):
        """
        There is no neural net to initialize, but we do need to initialize the
        parent class to inherit its user-defined functions.

        Args:
            gamma: The discount factor that will be used for defining the
            qualities that the AI outputs.
            r_b: The immediate reward for a turn where the monkey gets a banana
            r_e: The immediate reward for a turn where a monkey just eats food.
            r_d: The immediate reward of death.
        """
        BrainDQN.__init__(self)

        self.pi = self.pi_greedy

        self.gamma = gamma

        self.r_b = r_b
        self.r_e = r_e
        self.r_d = r_d

        self.Q_null = r_e*1/(1-gamma)
        self.Q_ave = (r_b + 3*r_e) * (1+gamma+gamma**2+gamma**3) / 4 + \
            gamma**4 * self.Q_null

        # Good estimate for the standard deviation of qualities
        self.Q_sd = (self.Q_ave-self.Q_null)/2

        self.save_Q = save_Q
        self.noisy = True

    def pi(self, s):# Set default policy.
        return self.pi_greedy(s)

    def __get_move__(self, channels):
        """
        This function finds the closest banana with a valid path.

        Args:
            channels: The channel map of the monkey's vision.
        
        Returns:
            1: Integer denoting number of moves away.
            2: Integer denoting the first move in the series.
        """
        banana_locations = channels[gl.INDEX_BANANA].nonzero()
        banana_reorigin = banana_locations - gl.RADIUS
        banana_distances = [torch.norm(x.float(), 1) for x in banana_reorigin]
        banana_sort = list(zip(banana_distances, range(len(banana_locations)), \
            banana_locations))
        banana_sort.sort()


        for banana_distance, uniquifier, banana_location in banana_sort:
            has_path, path = self.__has_path__(channels, banana_location)
            if has_path and len(path) > 0:
                return len(path), path[0]
        return 0, -1


    def __has_path__(self, channels, location):
        """
        This function determines if there is a path towards a banana.
        For this path, every move must be made walking towards the banana.

        Args:
            channels: The channel map of the monkey's vision.
            location: The location of the banana

        Returns:
            0: Boolean
            1: Tuple of integers denoting path.
        """

        relative_location = location-gl.RADIUS
        relative_location = relative_location.tolist()
        moves = []
        if relative_location[0] < 0:
            moves += [gl.WASD.index('w')]*abs(relative_location[0])
        else:
            moves += [gl.WASD.index('s')]*relative_location[0]
        if relative_location[1] < 0:
            moves += [gl.WASD.index('a')]*abs(relative_location[1])
        else:
            moves += [gl.WASD.index('d')]*relative_location[1]
        for perm in itertools.permutations(moves):
            if self.__valid_path__(channels, location, perm):
                return True, tuple(perm)
        return False, tuple()


    def __valid_path__(self, channels, location, path):
        """
        Determines if the path leads the monkey to the location without going
        through any walls or lava.

        Args:
            channels: The channel map of the monkey's vision.
            location: The location of the banana
            path: A tuple of moves to get to the location.

        Returns:
        0: Boolean
        """
        pos = (gl.RADIUS, gl.RADIUS)
        for action in path:
            symbol = gl.WASD[action]
            if symbol  == 'a':
                pos = (pos[0], pos[1]-1)
            elif symbol  == 'd':
                pos = (pos[0], pos[1]+1)
            elif symbol  == ' ':
                pos = (pos[0], pos[1])
            elif symbol  == 'w':
                pos = (pos[0]-1, pos[1])
            elif symbol  == 's':
                pos = (pos[0]+1, pos[1])
            if all(channels[[gl.INDEX_DANGER,gl.INDEX_BARRIER], \
                pos[0], pos[1]] == 0):
                # The way is clear:
                pass
            else:
                return False
        return True


    def __quality_event__(self, immediate_reward, turn):
        """
        Returns the quality of a sequence of moves where the monkey gets
        immediate_reward on turn "turn" and then just gets the average
        reward afterwards.

        Args:
            immediate_reward: The immediate reward of the event in question.
            turn: The turn to get that reward on.

        Returns:
            0: The quality of this sequence.
        """
        Q = immediate_reward + self.gamma * self.Q_ave
        for i in range(turn-1):
            Q = self.r_e + self.gamma * Q
        return Q

    def __noise__(self, sd):
        """
        Generate random noise in the qualities.

        Args:
            The number of standard deviations of noise to add.

        Returns:
            0: A vector of the same length of a quality of pure noise.
        """
        # Generate noise from the standard normal distribution
        noise = torch.randn(len(gl.WASD))
        # Rescale to standard deviation.
        noise *= self.Q_sd*sd
        return noise

    def forward(self, s):
        """
        This function generates a vector of qualities from the state given.

        Args:
            s: The state of the system.
        
        Returns:
            0: 5-tensor of qualities.
        """
        # Unpack state
        food, vision = s

        # First we need to get a move for the monkey
        distance, move = self.__get_move__(vision)

        # Now we know all the qualities must be lower than this,
        # so we can assign the qualities.
        # First the case where a banana was found:
        if distance > 0:
            Q = torch.zeros(len(gl.WASD)) + \
                self.__quality_event__(self.r_e, distance)
            Q[move] = self.__quality_event__(self.r_b, distance)
        else:
            Q = torch.zeros(len(gl.WASD)) + self.Q_null

        # Now we check for lava and barriers
        for action in range(len(gl.WASD)):
            pos = (gl.RADIUS, gl.RADIUS)
            symbol = gl.WASD[action]
            if symbol  == 'a':
                pos = (pos[0], pos[1]-1)
            elif symbol  == 'd':
                pos = (pos[0], pos[1]+1)
            elif symbol  == ' ':
                pos = (pos[0], pos[1])
            elif symbol  == 'w':
                pos = (pos[0]-1, pos[1])
            elif symbol  == 's':
                pos = (pos[0]+1, pos[1])
            if vision[gl.INDEX_DANGER, pos[0], pos[1]] > 0:
                Q[action] = self.r_d
            if vision[gl.INDEX_BARRIER, pos[0], pos[1]] > 0:
                Q[action] = self.Q_null

        # Finally, penalize staying still.
        Q[gl.WASD.index(' ')] = self.Q_null

        if vision[gl.INDEX_DANGER, gl.RADIUS, gl.RADIUS] > 0:
                Q[action] = self.r_d

        # Now we add some gaussian noise to it
        if self.noisy:
            Q += self.__noise__(0.05)

        if self.save_Q:
            self.last_Q = Q

        return Q
