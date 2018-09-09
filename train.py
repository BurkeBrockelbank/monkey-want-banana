"""
This is the trainer module. It includes functions for generating training data and
performing supervised training on the brain.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/trainer.py
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

import global_variables as gl
import exceptions
import room_generator as rg
import custom_loss
import monkey
import grid

import math
import random
import bisect

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import winsound

class PlateauFlagger:
    """
    This class keeps track of validation losses and returns a boolean
    every step to say whether it has plateaued.
    """
    def __init__(self, patience):
        self.patience = patience
        self.epochs = 0
        self.bad_epochs = 0
        self.best_loss = None

    def step(self, val_loss):
        """
        Returns:
            0: Boolean. True if the validation has plateaued, False otherwise
        """
        self.epochs += 1
        if self.epochs == 1:
            self.best_loss = val_loss
            return False

        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return False
        
        elif self.bad_epochs > self.patience:
            return True

        else:
            self.bad_epochs += 1
            return True


def monkey_training_data_1_life(N, life_span, path_root, g, loud = []):
    """
    This generates black-box style training data, but every life of the monkey
    is in only one data file. This makes it possible to generate better
    qualities. Single monkey tracked for data.

    Args:
        N: The number of turns that are allowed to take place in total.
        life_span: The number of turns allowable in a single life.
        path_root: The path to use for the saves. Integers are appended to
            the end for multiple files.
        g: The grid to generate training data from.
        loud: Default []. List of monkey indeces to watch.
    """
    life = 0
    life_turns = 0
    for total_turns in range(N):
        life_turns += 1
        if total_turns%100 == 0:
            print('Turn', total_turns, 'life', life)
        # Tick the monkeys
        foods, actions, surroundings = g.tick(0, invincible = True, \
            loud = loud)
        # Only record the first monkey
        food = foods[0]
        action = actions[0]
        surr = surroundings[0]
        outF = open(path_root+str(life)+'.dat', 'a')
        outF.write('(')
        outF.write(str(food))
        outF.write(',')
        outF.write(str(action.item()))
        outF.write(',')
        surr_string = str(surr)
        surr_string = surr_string.replace('tensor','torch.tensor')
        surr_string = surr_string.replace(' ','')
        surr_string = surr_string.replace('\n','')
        outF.write(surr_string)
        outF.write(')')
        outF.write('\n')
        outF.close()

        # Check for changing the file
        monkey = g.monkeys[0]
        if monkey.dead or life_turns == life_span:
            life += 1
            life_turns = 0
            monkey.dead = False
            monkey.food = monkey.start_food

            invalid_spot = True
            while invalid_spot:
                i = random.randrange(g.width)
                j = random.randrange(g.height)
                # This spot can have a monkey placed on it
                if g.channel_map[gl.INDEX_BARRIER,i,j] == 0 and \
                    g.channel_map[gl.INDEX_DANGER,i,j] == 0:
                    # First teleport the monkey on the channel map
                    g.teleport_monkey(monkey.pos, (i,j))
                    # Update the position in the monkey object
                    monkey.pos = (i,j)
                    # Flag for completion.
                    invalid_spot = False


def Q_training_data(N, paths, g, loud=[]):
    """
    This generates training data based on the actions of a monkey. The
    intention is for this to be used with an A.I.

    Args:
        N: The number of ticks in the training data.
        paths: A list of paths leading to the data files. One path must be
            present for each monkey in the grid.
        g: The grid to generate training data from.
        loud: Default []. List of monkey indeces to watch.
    """
    charity = False
    for n in range(N):
        if n%1000 == 0:
            print('Turn', n, 'begun.')
        # Tick the monkeys
        foods, actions, surroundings = g.tick(0, invincible = True, \
            loud=loud)
        # Iterate through the paths, surroundings, and actions
        for path, food, action, surr in zip(paths, foods, actions, surroundings):
            # Write the data to file
            outF = open(path, 'a')
            outF.write('(')
            outF.write(str(food))
            outF.write(',')
            outF.write(str(action.item()))
            outF.write(',')
            Q_surr_string = str(g.monkeys[0].brain.last_Q) + ',' + str(surr)
            Q_surr_string = Q_surr_string.replace('tensor','torch.tensor')
            Q_surr_string = Q_surr_string.replace(' ','')
            Q_surr_string = Q_surr_string.replace('\n','')
            outF.write(Q_surr_string)
            outF.write(')')
            outF.write('\n')
            outF.close()

        # If the monkey died:
        for monkey in g.monkeys:
            if monkey.dead:
                monkey.dead = False
                # If the monkey needs food, give it a few bananas.
                if monkey.food < 0:
                    monkey.eat(5)
                    # See if the monkey has starved to death before.
                    if charity:
                        # We should teleport the monkey to get it unstuck
                        charity = False
                        invalid_spot = True
                        while invalid_spot:
                            i = random.randrange(g.width)
                            j = random.randrange(g.height)
                            # This spot can have a monkey placed on it
                            if g.channel_map[gl.INDEX_BARRIER,i,j] == 0 and \
                                g.channel_map[gl.INDEX_DANGER,i,j] == 0:
                                # First teleport the monkey on the channel map
                                g.teleport_monkey(monkey.pos, (i,j))
                                # Update the position in the monkey object
                                monkey.pos = (i,j)
                                # Flag for completion.
                                invalid_spot = False
                    else:
                        charity = True

def monkey_training_data(N, paths, g, loud=[]):
    """
    This generates training data based on the actions of a monkey. The
    intention is for this to be used with an A.I.

    Args:
        N: The number of ticks in the training data.
        paths: A list of paths leading to the data files. One path must be
            present for each monkey in the grid.
        g: The grid to generate training data from.
        loud: Default []. List of monkey indeces to watch.
    """
    charity = False
    for n in range(N):
        print('Turn', n, 'begun.')
        # Tick the monkeys
        foods, actions, surroundings = g.tick(0, invincible = True, \
            loud=loud)
        # Iterate through the paths, surroundings, and actions
        for path, food, action, surr in zip(paths, foods, actions, surroundings):
            # Write the data to file
            outF = open(path, 'a')
            outF.write('(')
            outF.write(str(food))
            outF.write(',')
            outF.write(str(action.item()))
            outF.write(',')
            surr_string = str(surr)
            surr_string = surr_string.replace('tensor','torch.tensor')
            surr_string = surr_string.replace(' ','')
            surr_string = surr_string.replace('\n','')
            outF.write(surr_string)
            outF.write(')')
            outF.write('\n')
            outF.close()

        # If the monkey died:
        for monkey in g.monkeys:
            if monkey.dead:
                monkey.dead = False
                # If the monkey needs food, give it a few bananas.
                if monkey.food < 0:
                    monkey.eat(5)
                    # See if the monkey has starved to death before.
                    if charity:
                        # We should teleport the monkey to get it unstuck
                        charity = False
                        invalid_spot = True
                        while invalid_spot:
                            i = random.randrange(g.width)
                            j = random.randrange(g.height)
                            # This spot can have a monkey placed on it
                            if g.channel_map[gl.INDEX_BARRIER,i,j] == 0 and \
                                g.channel_map[gl.INDEX_DANGER,i,j] == 0:
                                # First teleport the monkey on the channel map
                                g.teleport_monkey(monkey.pos, (i,j))
                                # Update the position in the monkey object
                                monkey.pos = (i,j)
                                # Flag for completion.
                                invalid_spot = False
                    else:
                        charity = True

def training_data(N, paths, g,loud=[]):
    """
    This generates training data for the monkey with user input.
    
    Args:
        N: The number of ticks in the training data.
        paths: A list of paths leading to the data files. One path must be
            present for each monkey in the grid.
        g: The grid to generate training data from.
        loud: Default []. List of monkey indeces to watch.
    """
    for n in range(N):
        # Tick the monkeys
        foods, actions, surroundings = g.tick(2, loud=loud)
        # Iterate through the paths, surroundings, and actions
        for path, food, action, surr in zip(paths, foods, actions, surroundings):
            # Write the data to file
            outF = open(path, 'a')
            outF.write('(')
            outF.write(str(food))
            outF.write(',')
            outF.write(str(int(action)))
            outF.write(',')
            surr_string = str(surr)
            surr_string = surr_string.replace('tensor','torch.tensor')
            surr_string = surr_string.replace(' ','')
            surr_string = surr_string.replace('\n','')
            outF.write(surr_string)
            outF.write(')')
            outF.write('\n')
            outF.close()

def clean_data(in_paths, out_paths):
    """
    Removes portions of the data where the monkey is stuck.

    Args:
        in_paths: The paths pointing to the original data files.
        out_paths: The paths pointing to the target data files.
    """
    for in_path, out_path in zip(in_paths, out_paths):
        in_f = open(in_path, 'r')
        in_lines = in_f.readlines()
        in_f.close()
        # parse the input lines
        data = [eval(x.rstrip()) for x in in_lines]
        # Iterate backwards through data
        for i in range(len(data)-1, 3,-1):
            # Check if the surroundings are the same (infinite loops)
            # composed of either 1 or two states.
            if all(data[i][2].view(-1) == data[i-1][2].view(-1)) \
                or all(data[i][2].view(-1) == data[i-2][2].view(-1)) \
                or all(data[i][2].view(-1) == data[i-4][2].view(-1)):
                del data[i]
        out_f = open(out_path, 'w')
        for line in data:
            line = str(line)
            line = line.replace('tensor', 'torch.tensor')
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            out_f.write(line)
            out_f.write('\n')
        out_f.close()

def Q_supervised_training(epochs, batches, paths, brain, \
    lr, intermediate=''):
    """
    This performs supervised training on the monkey by consulting exact values
    of the qualities. Losses are reported every batch.

    
    Args:
        N: The number of epochs to run in training.
        paths: A list of paths leading to the data files.
        brain: The brain to train..
        lr: The learning rate to use.
        intermediate: The file to save intermediate brain trainings to.

    Returns:
        0: Training data in the form of list of tuples. First element is epoch
        number, second number is average loss over this epoch.
    """
    # Set the brain to training mode
    brain.train()

    data_set = []

    # First read all training data
    for path in paths:
        print('Reading', path)
        in_f = open(path, 'r')
        in_lines = in_f.readlines()
        in_f.close()
        # parse the input lines
        data = [eval(x.rstrip()) for x in in_lines]
        data_set += data

    # Calculate batch data
    batch_length = len(data_set)//batches

    # Report status
    print('Data loaded')

    # Now we do the actual learning!
    # Define the loss function
    criterion = nn.SmoothL1Loss()
    # Create an optimizer
    optimizer = torch.optim.Adam(brain.parameters(), lr=lr)
    loss_record = []
    # Iterate through epochs
    for epoch in range(epochs):
        # Permute the data to decorrelate it.
        random.shuffle(data_set)
        # Separate into batches
        batched_data = []
        for batch_no in range(batches-1):
            batch_start = batch_no*batch_length
            batched_data.append(data_set[batch_start:batch_start+batch_length])
        batched_data.append(data_set[(batches-1)*batch_length:])

        # Iterate through data
        for batch_no, batch_set in enumerate(batched_data):
            total_loss = 0
            print('Epoch', epoch, 'Batch', batch_no, 'begun')
            for food, action, qualities, vision in batch_set:
                s = (food, vision)
                # Get the quality of the action the monkey did
                predicted_Qs = brain.forward(s)
                # Calculate the loss
                loss = criterion(predicted_Qs[None],qualities[None])
                # Zero the gradients
                optimizer.zero_grad()
                # perform a backward pass
                loss.backward()
                # Update the weights
                optimizer.step()
                # Add to total loss
                total_loss += float(loss)
            # Add to loss record
            loss_record.append((epoch*batches+batch_no, total_loss/batch_length))
            print('Epoch', epoch, 'batch', batch_no, 'loss', total_loss/batch_length)

        # Save brain
        if intermediate != '':
            torch.save(brain.state_dict(), intermediate)

    return loss_record

def cross_entropy_supervised_training(epochs, batches, paths, brain, \
    lr, intermediate=''):
    """
    This performs supervised training on the monkey without consulting the
    exact values of the qualities. Qualities out of the model go through a
    softmax before being compared to a one-hot vector denoting the direction
    the AI moved. Due to all the cuts in the data, it has become unreliable
    to consult the quality. Losses are reported every batch.

    
    Args:
        N: The number of epochs to run in training.
        paths: A list of paths leading to the data files.
        brain: The brain to train..
        lr: The learning rate to use.
        intermediate: The file to save intermediate brain trainings to.

    Returns:
        0: Training data in the form of list of tuples. First element is epoch
        number, second number is average loss over this epoch.
    """
    # Set the brain to training mode
    brain.train()

    data_set = []

    # First read all training data
    for path in paths:
        print('Reading', path)
        in_f = open(path, 'r')
        in_lines = in_f.readlines()
        in_f.close()
        # parse the input lines
        data = [eval(x.rstrip()) for x in in_lines]
        data_set += data

    # Calculate batch data
    batch_length = len(data_set)//batches

    # Report status
    print('Data loaded')

    # Now we do the actual learning!
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Create an optimizer
    optimizer = torch.optim.Adagrad(brain.parameters(), lr=lr)
    loss_record = []
    # Iterate through epochs
    for epoch in range(epochs):
        # Permute the data to decorrelate it.
        random.shuffle(data_set)
        # Separate into batches
        batched_data = []
        for batch_no in range(batches-1):
            batch_start = batch_no*batch_length
            batched_data.append(data_set[batch_start:batch_start+batch_length])
        batched_data.append(data_set[(batches-1)*batch_length:])

        # Iterate through data
        for batch_no, batch_set in enumerate(batched_data):
            total_loss = 0
            print('Epoch', epoch, 'Batch', batch_no, 'begun')
            for food, action, vision in batch_set:
                s = (food, vision)
                # Get the quality of the action the monkey did
                predicted_Qs = brain.forward(s)
                # Calculate the loss
                loss = criterion(predicted_Qs[None], torch.LongTensor([action]))
                # Zero the gradients
                optimizer.zero_grad()
                # perform a backward pass
                loss.backward()
                # Update the weights
                optimizer.step()
                # Add to total loss
                total_loss += float(loss)
            # Add to loss record
            loss_record.append((epoch*batches+batch_no, total_loss/batch_length))
            print('Epoch', epoch, 'batch', batch_no, 'loss', total_loss/batch_length)

        # Save brain
        if intermediate != '':
            torch.save(brain.state_dict(), intermediate)

    return loss_record

def supervised_training(epochs, batches, paths, brain, gamma, \
    max_discount, lr, report = True, intermediate = ''):
    """
    This performs supervised training on the monkey. 
    
    Args:
        N: The number of epochs to run in training.
        paths: A list of paths leading to the data files.
        brain: The brain to train.
        gamma: The discount factor in the Bellman equation.
        max_discount: The maximum factor to allow for discount in calculating
        qualities.
        lr: The learning rate to use.
        reports: The number of times to print progress.
        intermediate: The file to save intermediate brain trainings to.


    Returns:
        0: Training data in the form of list of tuples. First element is epoch
        number, second number is average loss over this epoch.
    """
    # Set the brain to training mode
    brain.train()

    all_data = []

    # First read all training data
    for path in paths:
        print('Reading', path)
        in_f = open(path, 'r')
        in_lines = in_f.readlines()
        in_f.close()
        # parse the input lines
        data = [eval(x.rstrip()) for x in in_lines]
        # As a reminder, the data structure is
        # food (int), action (int),board state (torch.tensor dtype=torch.uint8)
        # Now we need to calculate the quality for each of these
        food_vals = [x[0] for x in data]
        # We now will subtract subsequent food values to get the change in food
        food_diffs = [food_vals[i]-food_vals[i-1] for i in \
            range(1,len(food_vals))]
        # Delete the final row of data because it has no food difference
        # that can be calculated 
        new_data = data[:-1]
        # Calculate qualities
        quals = [0]
        for food_diff in food_diffs[::-1]:
            quals.append(quals[-1]*gamma+food_diff)
        quals = quals[1:]
        quals = quals[::-1]
        # Insert the quality into the data
        new_data = [(torch.tensor(quality),) + state_tuple \
            for state_tuple, quality in zip(new_data, quals)]
        # Add to the list of data sets
        all_data.append(new_data)
    # Since the final quality values concatenate the series short, we should
    # cut those data points. We will arbitrarily decide to ignore rewards which
    # have a reduction in magnitute by the factor max_discount.
    n_to_cut = math.ceil(math.log(max_discount)/math.log(gamma))
    all_data = [x[:-n_to_cut] for x in all_data]
    # And now we have processed the data

    # Concatenate the data sets.
    data_set = [el for one_path in all_data for el in one_path]

    # Calculate batch data
    batch_length = len(data_set)//batches

    # Due to symmetry, we can increase the data set eightfold.
    data_symmetric = []
    # Still unimplemented

    # Report status
    print('Data loaded')

    # Now we do the actual learning!
    # Define the loss function
    criterion = custom_loss.L1ClampLoss
    # Create an optimizer
    optimizer = torch.optim.Adagrad(brain.parameters(), lr=lr)
    loss_record = []
    # Iterate through epochs
    for epoch in range(epochs):
        # Permute the data to decorrelate it.
        random.shuffle(data_set)
        # Separate into batches
        batched_data = []
        for batch_no in range(batches-1):
            batch_start = batch_no*batch_length
            batched_data.append(data_set[batch_start:batch_start+batch_length])
        batched_data.append(data_set[(batches-1)*batch_length:])

        # Iterate through data
        for batch_no, batch_set in enumerate(batched_data):
            total_loss = 0
            for real_Q, food, action, vision in batch_set:
                s = (food, vision)
                # Get the qualities the monkey deducts.
                predicted_Q = brain.forward(s)
                # Calculate the loss
                loss = criterion(predicted_Q, real_Q, action)
                # if loss > 1000:
                #     # There is some issue with the network occasionally spitting
                #     # out huge values. We will cap the maximum value. This is
                #     # done by recalculating the loss with something designed to
                #     # just be just 1000 away from the prediction. To get this
                #     # value, we need to pull the value from predicted_Q and
                #     # remove its needs_gradient property. This is done by casting
                #     # to a floating point number.
                #     raise RuntimeWarning('Loss has been calculated as ridiculous.')
                #     loss = criterion(predicted_Q, \
                #         torch.FloatTensor(float(predicted_Q)-1000))
                # Zero the gradients
                optimizer.zero_grad()
                # perform a backward pass
                loss.backward()
                # Update the weights
                optimizer.step()
                # Add to total loss
                if report:
                    total_loss += float(loss)
            # Add to loss record
            if report:
                loss_record.append((epoch*batches+batch_no, total_loss/batch_length))
                print('Epoch', epoch, 'batch', batch_no, 'loss', total_loss/batch_length)

        # Save brain
        if intermediate != '':
            torch.save(brain.state_dict(), intermediate)

    return loss_record

def load_data_paths(paths, gamma, max_discount=-1):
    """
    This loads in data paths and claculates gamma for each data point with a
    truncation error defined by max_discount.

    Args:
        paths: A list of paths leading to the data files.
        gamma: The discount factor in the Bellman equation.
        max_discount: Default -1. If specified, this defines the maximum discount
            that is considered before the Bellman series is truncated.
    """
    all_data = []
    for path in paths:
        in_f = open(path, 'r')
        in_lines = in_f.readlines()
        in_f.close()
        # parse the input lines
        data = [eval(x.rstrip()) for x in in_lines]
        # As a reminder, the data structure is
        # food (int), action (int),board state (torch.tensor dtype=torch.uint8)
        # Now we need to calculate the quality for each of these
        food_vals = [x[0] for x in data]
        # We now will subtract subsequent food values to get the change in food
        food_diffs = [food_vals[i]-food_vals[i-1] for i in \
            range(1,len(food_vals))]
        # Delete the final row of data because it has no food difference
        # that can be calculated 
        new_data = data[:-1]
        # Calculate qualities
        quals = [0]
        for food_diff in food_diffs[::-1]:
            quals.append(quals[-1]*gamma+food_diff)
        quals = quals[1:]
        quals = quals[::-1]
        # Insert the quality into the data
        new_data = [(torch.tensor(quality),) + state_tuple \
            for state_tuple, quality in zip(new_data, quals)]
        # Add to the list of data sets
        all_data.append(new_data)
    # Since the final quality values concatenate the series short, we should
    # cut those data points. We will arbitrarily decide to ignore rewards which
    # have a reduction in magnitute by the factor max_discount.
    if max_discount != -1:
        n_to_cut = math.ceil(math.log(max_discount)/math.log(gamma))
        all_data = [x[:-n_to_cut] for x in all_data]
    # Concatenate the data sets.
    data_set = [el for one_path in all_data for el in one_path]

    return data_set

def split_batches(data_set, batches, batch_length):
    """
    This function splits a data set into batches. Extra points are added to the final batch.

    Args:
        data_set: The data set to split.
        batches: The number of batches to do.
        batch_length: The length of a batch.

    Returns:
        0: Iterable of all the batches.
    """
    for batch_no in range(batches-1):
        batch_start = batch_no*batch_length
        yield data_set[batch_start:batch_start+batch_length]
    # The final batch gets the remaining points (less than number of batches)
    yield data_set[(batches-1)*batch_length:]

def plot_grid_record(plot_batch, plot_lr, plot_training, plot_val, \
    plot_test, plot_score, save_dir):
    """
    Plots a grid record and saves it to save_path.

    Args:

    """
    # Everything should have the same scale, so find the min and max
    vmin = min(plot_training.min(), plot_val.min(), plot_test.min())
    vmax = max(plot_training.max(), plot_val.max(), plot_test.max())
    levels = np.linspace(vmin,vmax,20)

    plt.clf()

    # Training set
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Training')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    pcm = ax.contourf(plot_batch, plot_lr, plot_training, \
        levels=levels, cmap=plt.get_cmap('viridis'))
    plt.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.15, \
        extend='both')
    plt.savefig(save_dir+'train.png')
    plt.clf()

    # Validation set
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Validation')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    pcm = ax.contourf(plot_batch, plot_lr, plot_val, \
        levels=levels, cmap=plt.get_cmap('viridis'))
    plt.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.15, \
        extend='both')
    plt.savefig(save_dir+'val.png')
    plt.clf()

    # Test set
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Test')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    pcm = ax.contourf(plot_batch, plot_lr, plot_test, \
        levels=levels, cmap=plt.get_cmap('viridis'))
    plt.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.15, \
        extend='both')
    plt.savefig(save_dir+'test.png')
    plt.clf()

    # Score
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Score')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    pcm = ax.contourf(plot_batch, plot_lr, plot_score, \
        cmap=plt.get_cmap('cubehelix'))
    plt.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.15, \
        extend='both')
    plt.savefig(save_dir+'score.png')
    plt.clf()



def grid_search_supervised(brain_class, max_epochs, batch_range, lr_range, paths, \
    gamma, scoring_room, data_dir,  max_discount=-1):
    """
    This performs a grid search on the number of batches to have and the learning
    rate to use.

    Args:
        max_epochs: The maximum number of epochs that are allowed. Training is halted
            when the loss on the training set drops below 80% of the loss on the
            validation set.
        batch_range: Tuple (min, max, N) of the minimum and maximum number of batches
            to test and how many batch numbers to test. Changed linearly.
        lr_range: Tuple (min, max, N) of the minimum and maximum learning rate to
            test and how many learning rates to test. Changed logarithmically.
        paths: The paths to the data.
        brain_class: Since the brain is recreated several times, just pass the
            class in and it will be instantiated here.
        paths: A list of paths leading to the data files.
        gamma: The discount factor in the Bellman equation.
        scoring_room: The room in which to calculate the score of the monkey.
        data_dir: The directory where plots and data are saved.
        max_discount: Default -1. If specified, this defines the maximum discount
            that is considered before the Bellman series is truncated.

    Returns:

    """
    # Define colors
    c_train = 'blue'
    c_val = 'orange'
    # First build the grid of hyperparameters
    print('Building hyperparameters...')
    batch_step = (batch_range[1]-batch_range[0])//batch_range[2]
    batch_grid = torch.arange(batch_range[0], batch_range[1], batch_step, \
        dtype = torch.int).tolist()
    lr_grid = torch.logspace(*lr_range).tolist()

    # Read in the data
    print('Reading data...')
    data_set = load_data_paths(paths, gamma, max_discount)

    # Split off the validation and test sets with 60/20/20 split
    percent_20_length = len(data_set)//5
    validation_set = data_set[:percent_20_length]
    data_set = data_set[percent_20_length:]
    test_set = data_set[:percent_20_length]
    data_set = data_set[percent_20_length:]
    print('training set:', len(data_set))
    print('validation set:', len(validation_set))
    print('test set:', len(test_set))
    print('total:', len(data_set)+len(validation_set)+len(test_set))

    print('Batches', batch_grid)
    print('Batch sizes', [len(data_set)/x for x in batch_grid])
    print('lr', lr_grid)

    # Define the loss criterion
    criterion = custom_loss.L1_clamp_loss

    # Record for losses for the grid search
    grid_record = []
    plot_batch = np.zeros((len(batch_grid), len(lr_grid)), dtype=np.int_)
    plot_lr = np.zeros_like(plot_batch, dtype=float)
    plot_training = np.zeros_like(plot_lr)
    plot_val = np.zeros_like(plot_lr)
    plot_test = np.zeros_like(plot_lr)
    plot_score = np.zeros_like(plot_lr)

    print('Training nets...')

    # Now iterate through the grid
    for batch_index, batches in enumerate(batch_grid):
        # Calculate batch data
        batch_length = len(data_set)//batches
        for lr_index, lr in enumerate(lr_grid):
            # Create record for training and validation loss over epochs
            grid_point_record = []

            # Instantiate brain
            brain = brain_class()
            brain.pi = brain.pi_greedy
            brain.train()

            # Create optimizer and scheduler
            optimizer = torch.optim.Adam(brain.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                verbose = True, patience = 5)
            flagger = PlateauFlagger(10)

            # Run through epochs.
            epoch = 0
            overfit = False
            stagnant = False
            while epoch < max_epochs and not overfit and not stagnant:
                # Permute the data to decorrelate it.
                random.shuffle(data_set)
                # Separate into batches
                batched_data = list(split_batches(data_set, batches, batch_length))

                # Iterate through data
                total_loss = 0
                for batch_no, batch_set in enumerate(batched_data):
                    column_data = list(zip(*batch_set))
                    real_Q_vec = torch.stack(column_data[0])
                    foods = torch.tensor(column_data[1])
                    actions = torch.tensor(column_data[2])
                    visions = torch.stack(column_data[3])
                    predicted_Qs = brain.forward(foods, visions)
                    loss = criterion(predicted_Qs, real_Q_vec, actions).mean()
                    # print('prediction', predicted_Qs[0])
                    # print('Q(s,', gl.WASD[actions[0]], ') = ', real_Q_vec[0], sep='')
                    # Zero the gradients
                    optimizer.zero_grad()
                    # perform a backward pass
                    loss.backward()
                    # Update the weights
                    optimizer.step()
                    # Add to loss record
                    total_loss += loss.item()
                training_loss = total_loss/batches

                # Validata data
                brain.eval()
                column_data = list(zip(*validation_set))
                real_Q_vec = torch.stack(column_data[0])
                foods = torch.tensor(column_data[1])
                actions = torch.tensor(column_data[2])
                visions = torch.stack(column_data[3])
                predicted_Qs = brain.forward(foods, visions)
                loss = criterion(predicted_Qs, real_Q_vec, actions).mean()
                val_loss = loss.item()
                brain.train()

                # Add to record
                grid_point_record.append((epoch, training_loss, val_loss))

                # Update learning rate
                scheduler.step(val_loss)

                # Check for plateau
                if flagger.step(val_loss):
                    stagnant = True

                # Check for overfitting
                elif training_loss/val_loss < 0.6:
                    overfit = True

                # Update epoch
                epoch += 1
            print(batches, lr, 'ran with', epoch, 'epochs')

            # Calculate loss on test set
            brain.eval()
            column_data = list(zip(*test_set))
            real_Q_vec = torch.stack(column_data[0])
            foods = torch.tensor(column_data[1])
            actions = torch.tensor(column_data[2])
            visions = torch.stack(column_data[3])
            predicted_Qs = brain.forward(foods, visions)
            loss = criterion(predicted_Qs, real_Q_vec, actions).mean()
            test_loss = loss.item()
            brain.train()

            # Calculate score
            brain.eval()
            g = grid.Grid([monkey.Monkey(brain)], scoring_room)
            score, score_err = test_model(g, 50, 30, loud=False)
            brain.train()

            # Save the data for this grid point
            grid_record.append((batches, lr, training_loss, val_loss, test_loss, score))

            # Save data in plotting data structure
            plot_batch[batch_index, lr_index] = batch_length
            plot_lr[batch_index, lr_index] = lr
            plot_training[batch_index, lr_index] = training_loss
            plot_val[batch_index, lr_index] = val_loss
            plot_test[batch_index, lr_index] = test_loss
            plot_score[batch_index, lr_index] = score

            # Save training record
            point_path = data_dir + 'batch' + str(batches) + 'lr' + str(lr)
            epoch_data = [x[0] for x in grid_point_record]
            training_epoch_data = [x[1] for x in grid_point_record]
            validation_epoch_data = [x[2] for x in grid_point_record]
            plt.plot(training_epoch_data, c=c_train)
            plt.plot(validation_epoch_data, c=c_val)
            plt.savefig(point_path + '.png')
            plt.clf()
            out_f = open(point_path+'.dat','w')
            for epoch, training_loss, val_loss in grid_point_record:
                out_f.write(str(epoch))
                out_f.write(' ')
                out_f.write(str(training_loss))
                out_f.write(' ')
                out_f.write(str(val_loss))
                out_f.write('\n')
            out_f.close()

            # Save the brain at the end of training
            torch.save(brain.state_dict(), point_path + '.brainsave')

    print('Finishing up...')
    # Write the data for the grid search
    out_f = open(data_dir+'search.dat', 'w')
    for batches, lr, training_loss, val_loss, test_loss, score in grid_record:
        out_f.write(str(batches))
        out_f.write(' ')
        out_f.write(str(lr))
        out_f.write(' ')
        out_f.write(str(training_loss))
        out_f.write(' ')
        out_f.write(str(val_loss))
        out_f.write(' ')
        out_f.write(str(test_loss))
        out_f.write(' ')
        out_f.write(str(score))
        out_f.write('\n')
    out_f.close()

    # Generate plots for search
    plot_grid_record(plot_batch, plot_lr, plot_training, plot_val, \
        plot_test, plot_score, data_dir)

    # plt.figure().add_subplot(111, projection='3d')\
    #     .plot_wireframe(plot_batch, plot_lr, plot_training, \
    #         colors = [c_train])
    # plt.savefig(data_dir+'training.png')
    # plt.close()

    # plt.figure().add_subplot(111, projection='3d')\
    #     .plot_wireframe(plot_batch, plot_lr, plot_val, \
    #         colors = [c_val])
    # plt.savefig(data_dir+'val.png')
    # plt.close()

    # plt.figure().add_subplot(111, projection='3d')\
    #     .plot_wireframe(plot_batch, plot_lr, plot_test, \
    #         colors = [c_test])
    # plt.savefig(data_dir+'test.png')
    # plt.close()

    # plt.figure().add_subplot(111, projection='3d')\
    #     .plot_wireframe(plot_batch, plot_lr, plot_training, \
    #         colors = [c_train])
    # plt.figure().add_subplot(111, projection='3d')\
    #     .plot_wireframe(plot_batch, plot_lr, plot_val, \
    #         colors = [c_val])
    # plt.figure().add_subplot(111, projection='3d')\
    #     .plot_wireframe(plot_batch, plot_lr, plot_test, \
    #         colors = [c_test])
    # plt.savefig(data_dir+'all.png')
    # plt.close()




def supervised_columns(epochs, batches, paths, brain, gamma, \
    lr, max_discount=-1, report = True, intermediate = ''):
    """
    This performs supervised training on the monkey. First path is assumed to
    point to the validation set.
    
    Args:
        N: The number of epochs to run in training.
        paths: A list of paths leading to the data files.
        brain: The brain to train.
        gamma: The discount factor in the Bellman equation.
        max_discount: Default -1. If otherwise specified, this defines the highest
            discount to be considered in the series.
        lr: The learning rate to use.
        reports: The number of times to print progress.
        intermediate: The file to save intermediate brain trainings to.
        validation_index: Which file to use for the validation set


    Returns:
        0: Training data in the form of list of tuples. First element is epoch
        number, second number is average loss over this epoch.
    """
    # Set the brain to training mode
    brain.train()

    all_data = []

    # First read all training data
    for path in paths:
        print('Reading', path)
        in_f = open(path, 'r')
        in_lines = in_f.readlines()
        in_f.close()
        # parse the input lines
        data = [eval(x.rstrip()) for x in in_lines]
        # As a reminder, the data structure is
        # food (int), action (int),board state (torch.tensor dtype=torch.uint8)
        # Now we need to calculate the quality for each of these
        food_vals = [x[0] for x in data]
        # We now will subtract subsequent food values to get the change in food
        food_diffs = [food_vals[i]-food_vals[i-1] for i in \
            range(1,len(food_vals))]
        # Delete the final row of data because it has no food difference
        # that can be calculated 
        new_data = data[:-1]
        # Calculate qualities
        quals = [-1.31]
        for food_diff in food_diffs[::-1]:
            quals.append(quals[-1]*gamma+food_diff)
        quals = quals[1:]
        quals = quals[::-1]
        # Insert the quality into the data
        new_data = [(torch.tensor(quality),) + state_tuple \
            for state_tuple, quality in zip(new_data, quals)]
        # Add to the list of data sets
        all_data.append(new_data)
    # Since the final quality values concatenate the series short, we should
    # cut those data points. We will arbitrarily decide to ignore rewards which
    # have a reduction in magnitute by the factor max_discount.
    if max_discount != -1:
        n_to_cut = math.ceil(math.log(max_discount)/math.log(gamma))
        all_data = [x[:-n_to_cut] for x in all_data]
    # And now we have processed the data

    # Concatenate the data sets.
    data_set = [el for one_path in all_data for el in one_path]

    # Shuffle the data set
    random.shuffle(data_set)
    # Pull out the validation set
    validation_length = len(data_set)//5
    validation_set = data_set[:validation_length]
    data_set = data_set[validation_length:]

    # Calculate batch data
    batch_length = len(data_set)//batches

    # Report status
    print('Data loaded')

    # Now we do the actual learning!
    # Define the loss function
    criterion = custom_loss.L1_clamp_loss
    # Create an optimizer
    optimizer = torch.optim.Adagrad(brain.parameters(), lr=lr)
    # Learning rate decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
        factor=0.5, verbose = report)
    loss_record = []
    if report:
        print(batches, 'batches of', batch_length, 'data points')
    # Iterate through epochs
    for epoch in range(epochs):
        total_loss = 0
        # Permute the data to decorrelate it.
        random.shuffle(data_set)
        # Separate into batches
        batched_data = []
        for batch_no in range(batches-1):
            batch_start = batch_no*batch_length
            batched_data.append(data_set[batch_start:batch_start+batch_length])
        # The final batch gets the remaining points (less than number of batches)
        batched_data.append(data_set[(batches-1)*batch_length:])

        # Iterate through data
        for batch_no, batch_set in enumerate(batched_data):
            column_data = list(zip(*batch_set))
            real_Q_vec = torch.stack(column_data[0])
            foods = torch.tensor(column_data[1])
            actions = torch.tensor(column_data[2])
            visions = torch.stack(column_data[3])
            predicted_Qs = brain.forward(foods, visions)
            loss = criterion(predicted_Qs, real_Q_vec, actions).mean()
            # print('prediction', predicted_Qs[0])
            # print('Q(s,', gl.WASD[actions[0]], ') = ', real_Q_vec[0], sep='')
            # Zero the gradients
            optimizer.zero_grad()
            # perform a backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            # Add to loss record
            total_loss += loss.item()

        # Save brain
        if intermediate != '':
            torch.save(brain.state_dict(), intermediate)

        # Validata data
        brain.eval()
        column_data = list(zip(*validation_set))
        real_Q_vec = torch.stack(column_data[0])
        foods = torch.tensor(column_data[1])
        actions = torch.tensor(column_data[2])
        visions = torch.stack(column_data[3])
        predicted_Qs = brain.forward(foods, visions)
        loss = criterion(predicted_Qs, real_Q_vec, actions).mean()
        val_loss = loss.item()
        brain.train()

        # Update learning rate
        scheduler.step(val_loss)


        if report:
            loss_record.append((epoch, total_loss/batches, val_loss))
            print('Epoch', epoch, 'loss', total_loss/batches, 'Validation loss', val_loss)

    return loss_record

def load_records(path):
    """
    Loads in the records for loss function vs. epochs
    Args:
        path: The path to the record file.
    Returns:
        0: A list of tuples of the form (epochs, loss)
    """
    records = []
    in_file = open(path, 'r')
    for line in in_file:
        records.append(eval(line.rstrip()))
    in_file.close()
    
    # Update the epoch numbers in the records
    for i in range(1, len(records)):
        start_epoch = records[i-1][-1][0]+1
        new_epochs = []
        for point in records[i]:
            new_epochs.append((point[0]+start_epoch, point[1]))
        records[i] = new_epochs
    # Join the records together
    if len(records) == 1:
        return records[0]
    else:
        return sum(records, [])


def curated_bananas_dqn(g, level, N, gamma, lr, food, random_start = False, \
    epsilon = lambda x: 0, watch = False, block_index = 2):
    """
    This function trains the monkey with curated examples. For example,
    at level 1, we will have a banana one block away from the monkey:
     b       b       b                                         
      m      m      m     bm     mb     m      m      m             
                                       b       b       b 
    Level two has the banana 2 or 1 away and so on.

    Args:
        g: The grid with a single monkey to train. Channel map will be
            rewritten.
        level: The level of curation to train.
        N: The number of iterations of training to do.
        lr: learning rate.
        food: The maximum food level. All food levels will be used randomly
            up to this level.
        random_start: This will make the first move of the monkey be
            completely random. Good for progressing on earlier curations.
        epsilon_data: Default lambda function returning zero. A function that
            calculates epsilon.
        watch: Default False. True if you want to watch the monkey train.
        block_index: The index of the type of block to be placed. Defaule 2 for
            banana.
    """
    # Set reporting up
    loud = []
    if watch:
        loud = [0]

    # Set monkey to train
    g.monkeys[0].brain.train()

    # Deal with level zero special case (only adjacent).
    level_zero = False
    if level == 0:
        level = 1
        level_zero = True

    # Build the channel map
    map_size = len(gl.SIGHT)+4*level
    radius = map_size//2
    g.channel_map = torch.zeros((len(gl.BLOCK_TYPES), map_size, map_size), \
        dtype = torch.uint8)
    # Build walls
    for i in range(map_size):
        for j in [0, -1]:
            # Place barrier
            g.channel_map[gl.INDEX_BARRIER, i, j] = 1
            # Place barrier in transpose position
            g.channel_map[gl.INDEX_BARRIER, j, i] = 1

    if level_zero:
        zero_positions = [(radius+1,radius), (radius-1,radius),
            (radius,radius+1), (radius,radius-1)]

    # Place monkey for new channel map
    g.monkeys[0].pos = (0,0)
    g.replace_monkeys()

    # Update width and height of channel map
    g.width = len(g.channel_map)
    g.height = len(g.channel_map[0])

    # Unpack epsilon if it exists
    epsilon_needed = False
    if g.monkeys[0].brain.pi == g.monkeys[0].brain.pi_epsilon_greedy:
        epsilon_needed = True

    # Initialize block position variables
    block_i = 0
    block_j = 0

    # Initialize loss record
    loss_record = []

    # Calculate the number of turns the monkey has to explore.
    turn_allowance = math.ceil((2*level**2 + 3*level + 1)/(2*level + 2))
    if level_zero:
        turn_allowance = 1

    # Report actions
    if level_zero:
        print("Curated training beginning. Level", 0, "allowing", \
            turn_allowance, "turns.")
    else:
        print("Curated training beginning. Level", level, "allowing", \
            turn_allowance, "turns.")

    for n in range(N):
        if n%100 == 0:
            print('Curated training', n, 'epsilon', epsilon(n))
        # Assign food
        g.monkeys[0].food = random.randrange(2*level,food)

        # Remove old blocks
        g.channel_map[block_index] = torch.zeros((map_size, map_size), \
            dtype = torch.uint8)

        # Assign block position
        position_chosen = False
        while not position_chosen:
            if not level_zero:
                block_i = random.randrange(radius-level, radius+level+1)
                block_j = random.randrange(radius-level, radius+level+1)
                if (block_i, block_j) != (radius, radius):
                    position_chosen = True
            else:
                block_i, block_j = random.choice(zero_positions)
                position_chosen = True
        g.channel_map[block_index, block_i, block_j] = 1

        # Replace monkey
        g.teleport_monkey(g.monkeys[0].pos, (radius,radius))
        g.monkeys[0].pos = (radius,radius)

        # Do a run of dqn training on monkey. The algorithm is slightly
        # different than in regular reinforcement learning. The monkey is
        # allowed to take a maximum of turn_allowance moves, but it will
        # stop short if the monkey gets the banana.
        # Pre loop:
        #   a) Calculate state
        #   b) Calculate policy
        # Every turn:
        #   a) Get policy action
        #   b) Get subsequent state
        #   c) Calculate immediate reward
        #   d) Get loss
        #   g) Step

        # a) Calculate state
        sight_new = g.surroundings(g.monkeys[0].pos)
        food_new = g.monkeys[0].food
        state_new = (food_new, sight_new)
        # b) Calculate policy
        # Do a random start if necessary
        if random_start:
            Q_new, a_new = g.monkeys[0].brain.pi_random(state_new)
            p_new = 0
            if epsilon_needed:
                epsilon_n = epsilon(n)
        elif epsilon_needed:
            epsilon_n = epsilon(n)
            Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new, epsilon_n)
        else:
            Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new)


        # Define optimizer
        optimizer = torch.optim.Adam(g.monkeys[0].brain.parameters(), lr=lr)
        # Define loss criterion
        criterion = nn.SmoothL1Loss(size_average=False)

        # Initialize something to end before the turn allowance if we
        # get to an end state
        end_state = False
        # Placeholder for total loss
        total_loss = 0
        # Iterate through all the allowable turns
        for turn_count in range(turn_allowance):
            if watch:
                print('-----------------------')

            # a) Get the policy's action.
            Q = Q_new
            a = a_new
            p = p_new

            # b) Get the consequent state (move the monkey).
            g.tick(1, directions = [a], invincible = True, loud=loud, wait=False)
            state_old = state_new
            sight_new = g.surroundings(g.monkeys[0].pos)
            food_new = g.monkeys[0].food
            state_new = (food_new, sight_new)

            # c) Get the immediate reward.
            # Immediate reward is normally food difference.
            r = state_new[0]-state_old[0]
            # If the monkey is dead, it instead gets a large penalty
            if g.monkeys[0].dead:
                r = gl.DEATH_REWARD
                g.monkeys[0].dead = False
                end_state = True

            # d) Calculate the loss
            # b) Calculate the maximum quality of the subsequent move
            if epsilon_needed:
                Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new, epsilon_n)
            else:
                Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new)
            # c) Calculate the loss difference
            delta = Q - r - gamma * Q_new
            # d) Calculate the loss as Huber loss.
            loss = criterion(delta, torch.zeros(1))
            total_loss += (float(loss))

            if watch:
                print(gl.WASD[a], 'with probability', p)
                print('Q(s,', gl.WASD[a], ') = ', round(float(Q),3), sep='')
                print('--> r + gamma * Q(s\',', gl.WASD[a_new], ')', sep='')
                print('  = ', r, ' + ', gamma, ' * ', round(float(Q_new),3), sep='')
                print('  = ', round(float(r+gamma*Q_new),3), sep='')
                print('delta = ' + str(float(delta)))
                input('loss = ' + str(float(loss)))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward(retain_graph= ((turn_count != turn_allowance-1) \
                and not end_state))
            optimizer.step()

            if end_state:
                break


        loss_record.append((n, total_loss))

        # Reset turn counter
        g.turn_count = 0

    return loss_record

def epsilon_interpolation(x,y):
    """
    This function returns a function that is a linear interpolation
    of the points given in the arguments.

    Args:
        x: The x values.
        y: The y values.

    Returns:
        0: Function
    """
    def out_func(p):
        if p > 100 or p < 0:
            raise ValueError('Epsilon interpolation takes a percentage between 0 and 100')
        elif p in x:
            return y[x.index(p)]
        else:
            bi = bisect.bisect(x, p)-1
            m = (y[bi+1]-y[bi])/(x[bi+1]-x[bi])
            return y[bi] + m*(p-x[bi])
    return out_func




def guided_dqn(g, test_g, N, gamma, lr, guide, epsilon_guide, epsilon_explore, watch=False):
    """
    Runs the DQN algorithm but instead of using pi_greedy or pi_epsilon greedy,
    a guiding AI is used to decide on moves.

    Args:
        g: The grid containing a single monkey containing a brain of
            superclass Brain_DQN.
        test_g: The grid to test on. Make sure the of the monkey in the test grid
            has the same instance of brain as the one in g
        N: The number of iterations of training to do.
        gamma: The discount for the Bellman equation.
        lr: The learning rate.
        guide: An instance of the brain class whose pi_greedy will be
            consulted.
        epsilon_guide: A function which converts a percentage to a value
            in [0,1]. The result is the chance of taking a move defined by the
            neural net.
        epsilon_explore: A function which converts a percentage to a value in
            [0,1]. The result is the chance of taking a move at random when the
            neural net has been chosen to take a move.
        watch: Default False. If True, will wait for the user to look at every
            iteration of the training.

    Returns:
        0: Training data in the form of list of tuples. First element is
        iteration number, second number is average loss over the
        iterations leading up to this report.
    """
    # Set monkey's pi to pi_greedy for scorekeeping
    g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_greedy

    # Put the training brain in the test grid monkey
    test_g.monkeys[0].brain = g.monkeys[0].brain

    # Determine if we want to watch
    if watch:
        loud = [0]
    else:
        loud = []

    # Instantiate total reward
    total_reward = 0

    # Calculate the state for the first time.
    g.monkeys[0].brain.eval()
    sight_new = g.surroundings(g.monkeys[0].pos)
    food_new = g.monkeys[0].food
    state_new = (food_new, sight_new)
    Q_guide, a_guide, p_guide = guide.pi_greedy(state_new)
    a_new = a_guide
    p_new = p_guide
    Q_new = g.monkeys[0].brain.Q(state_new, a_new)
    g.monkeys[0].brain.train()


    # Define optimizer
    optimizer = torch.optim.Adam(g.monkeys[0].brain.parameters(), lr=lr)
    # Define loss criterion
    criterion = nn.SmoothL1Loss(size_average=False)


    loss_record = []

    # Percentile reports
    one_percent = N//100
    if one_percent == 0:
        one_percent = N+1

    # new_pos = None ########
    # just_died = 0 ###########

    # Iterate N times
    for n in range(N):
        if n%one_percent == 0:
            # Test the monkey
            percentage = n//one_percent
            report_string = 'Learning is ' + str(percentage) + '% complete.'
            test_g.monkeys[0].brain.eval()
            test_result, test_err = test_model(test_g, 500, 30, loud=False)
            report_string += '. Score ' + str(test_result) +'(' + str(test_err) + ')'
            test_g.monkeys[0].brain.train()
            # Find the chance of the neural net making a decision
            epsilon_value = epsilon_guide(percentage)
            epsilon_random = epsilon_explore(percentage)

            print(report_string)

        if watch:
            print('-----------------------')

        # 1) Get the policy's action.
        Q = Q_new
        a = a_new
        p = p_new

        # 2) Get the consequent state (move the monkey).
        # old_monkey_pos = g.monkeys[0].pos #############
        g.tick(1, directions = [a], invincible = True, loud=loud, wait=False)
        state_old = state_new
        sight_new = g.surroundings(g.monkeys[0].pos)
        food_new = g.monkeys[0].food
        state_new = (food_new, sight_new)
        # old_pos = new_pos ##########

        # 3) Get the immediate reward.
        # Immediate reward is normally food difference.
        r = state_new[0]-state_old[0]
        # If the monkey is dead, it instead gets a large penalty
        # just_died += 1############
        if g.monkeys[0].dead:
            # just_died = -1############
            r = gl.DEATH_REWARD
            # If the monkey died of hunger, feed it.
            if g.monkeys[0].food < 0:
                g.monkeys[0].eat(5)
                # Teleport the monkey
                i, j = rg.free_spot(g.channel_map)
                g.teleport_monkey(g.monkeys[0].pos, (i,j))
                g.monkeys[0].pos = (i,j)
                # Rebuild the state
                sight_new = g.surroundings(g.monkeys[0].pos)
                state_new = (g.monkeys[0].food, sight_new)
            g.monkeys[0].dead = False
        total_reward += r

        # new_monkey_pos = g.monkeys[0].pos ##########

        # 4) Calculate the loss
        # a) Calculate the quality of the move undertaken
        # This was already done in part 1.
        # b) Calculate the maximum quality of the subsequent move
        if random.random() < epsilon_value:
            Q_guide, a_guide, p_guide = guide.pi_greedy(state_new)
            a_new = a_guide
            p_new = p_guide
            Q_new = g.monkeys[0].brain.Q(state_new, a_new)
        else:
            Q_new, a_new, p_new = \
            g.monkeys[0].brain.pi_epsilon_greedy(state_new, epsilon_random)
        # new_pos = (g.channel_map[1].nonzero()) ############
        # c) Calculate the loss difference
        delta = Q - r - gamma * Q_new
        # d) Calculate the loss as Huber loss.
        loss = criterion(delta, torch.zeros(1))
        # if abs(r) > 10:
        #     print('reward', r)
        #     print(state_new[0], state_old[0])
        #     input()
        loss_record.append((n,float(loss),test_result, test_err, epsilon_value, epsilon_random))

        # if loss > 12:
        #     winsound.Beep(440, 2000)
        #     print('Just died', just_died)
        #     print(old_pos, '-->', new_pos, '|', old_monkey_pos, '-->', new_monkey_pos)
        #     gen_states = (g.surroundings(old_monkey_pos), g.surroundings(new_monkey_pos))
        #     print('food', state_old[0], '-->', state_new[0])
        #     print(rg.channel_to_ASCII(state_old[1]))
        #     print('------------------')
        #     print(rg.channel_to_ASCII(state_new[1]))
        #     print('==================')
        #     print(rg.channel_to_ASCII(gen_states[0]))
        #     print('------------------')
        #     print(rg.channel_to_ASCII(gen_states[1]))
        #     print(gl.WASD[a], 'with probability', p)
        #     print('Q(s,', gl.WASD[a], ') = ', round(float(Q),3), sep='')
        #     print('--> r + gamma * Q(s\',', gl.WASD[a_new], ')', sep='')
        #     print('  = ', r, ' + ', gamma, ' * ', round(float(Q_new),3), sep='')
        #     print('  = ', round(float(r+gamma*Q_new),3), sep='')
        #     print('delta = ' + str(float(delta)))
        #     input('loss = ' + str(float(loss)))


        if watch:
            print(gl.WASD[a], 'with probability', p)
            print('Q(s,', gl.WASD[a], ') = ', round(float(Q),3), sep='')
            print('--> r + gamma * Q(s\',', gl.WASD[a_new], ')', sep='')
            print('  = ', r, ' + ', gamma, ' * ', round(float(Q_new),3), sep='')
            print('  = ', round(float(r+gamma*Q_new),3), sep='')
            print('delta = ' + str(float(delta)))
            input('loss = ' + str(float(loss)))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward(retain_graph= (n!=N-1))
        optimizer.step()

    return loss_record

def dqn_training(g, N, gamma, lr, \
    epsilon = lambda x: 0, watch = False):
    """
    This function trains a monkey with reinforcement learning.

    The DQN algorihm:
    1) Get the policy's action.
    2) Get the consequent state (move the monkey).
    3) Get the immediate reward from the grid.
    4) Calculate the loss
        a) Calculate the quality of the move undertaken Q(s,a).
        b) Calculate the max_a Q(s',a) where s' is the consequent
           state of performing a from the state s.
        c) delta = Q(s,a) - r - gamma*max_a Q(s', a)
           where r is the immediate loss measured from the system.
        d) Loss is the Huber loss (smooth L1 loss) of delta.

    Args:
        g: The grid containing a single monkey containing a brain of
            superclass Brain_DQN.
        N: The number of iterations of training to do.
        gamma: The discount for the Bellman equation.
        epsilon: Default lambda function returning zero. A function that gives
            the value for epsilon based on epoch number.
        lr: The learning rate.
        watch: Default False. If True, will wait for the user to look at every
            iteration of the training.

    Returns:
        0: Training data in the form of list of tuples. First element is
        iteration number, second number is average loss over the
        iterations leading up to this report.
    """
    # Determine if we want to watch
    if watch:
        loud = [0]
    else:
        loud = []

    # Unpack epsilon if it exists
    epsilon_needed = False
    if g.monkeys[0].brain.pi == g.monkeys[0].brain.pi_epsilon_greedy:
        epsilon_needed = True

    # Instantiate total reward
    total_reward = 0

    # Calculate the state for the first time.
    g.monkeys[0].brain.eval()
    sight_new = g.surroundings(g.monkeys[0].pos)
    food_new = g.monkeys[0].food
    state_new = (food_new, sight_new)
    if epsilon_needed:
        Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new, epsilon(0))
    else:
        Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new)
    g.monkeys[0].brain.train()


    # Define optimizer
    optimizer = torch.optim.Adam(g.monkeys[0].brain.parameters(), lr=lr)
    # Define loss criterion
    criterion = nn.SmoothL1Loss(size_average=False)


    loss_record = []

    # Percentile reports
    one_percent = N//100
    if one_percent == 0:
        one_percent = N+1

    # Iterate N times
    for n in range(N):
        if n%one_percent == 0:
            report_string = 'Learning is ' + str(n//one_percent) + \
                '% complete. epsilon = ' + str(round(epsilon(n),3))
            g.monkeys[0].brain.eval()
            g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_greedy
            test_result, test_err = test_model(g, 100, 30, loud=False)
            report_string += '. Score ' + str(test_result) +'('+str(test_err)+')'
            if epsilon_needed:
                g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_epsilon_greedy
            g.monkeys[0].brain.train()

            print(report_string)

        if watch:
            print('-----------------------')

        # 1) Get the policy's action.
        Q = Q_new
        a = a_new
        p = p_new

        # 2) Get the consequent state (move the monkey).
        g.tick(1, directions = [a], invincible = True, loud=loud, wait=False)
        state_old = state_new
        sight_new = g.surroundings(g.monkeys[0].pos)
        food_new = g.monkeys[0].food
        state_new = (food_new, sight_new)

        # 3) Get the immediate reward.
        # Immediate reward is normally food difference.
        r = state_new[0]-state_old[0]
        # If the monkey is dead, it instead gets a large penalty
        if g.monkeys[0].dead:
            r = gl.DEATH_REWARD
            # If the monkey died of hunger, feed it.
            if g.monkeys[0].food < 0:
                g.monkeys[0].eat(5)
                state_new = (g.monkeys[0].food, sight_new)
            g.monkeys[0].dead = False
        total_reward += r

        # 4) Calculate the loss
        # a) Calculate the quality of the move undertaken
        # This was already done in part 1.
        # b) Calculate the maximum quality of the subsequent move
        if epsilon_needed:
            Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new, epsilon(n))
        else:
            Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new)
        # c) Calculate the loss difference
        delta = Q - r - gamma * Q_new
        # d) Calculate the loss as Huber loss.
        loss = criterion(delta, torch.zeros(1))
        loss_record.append((n,float(loss),test_result))

        if watch:
            print(gl.WASD[a], 'with probability', p)
            print('Q(s,', gl.WASD[a], ') = ', round(float(Q),3), sep='')
            print('--> r + gamma * Q(s\',', gl.WASD[a_new], ')', sep='')
            print('  = ', r, ' + ', gamma, ' * ', round(float(Q_new),3), sep='')
            print('  = ', round(float(r+gamma*Q_new),3), sep='')
            print('delta = ' + str(float(delta)))
            input('loss = ' + str(float(loss)))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward(retain_graph= (n!=N-1))
        optimizer.step()

    return loss_record

def dqn_training_columns(brain, rooms, epochs, gamma, lr, \
    epsilon = lambda x: 0, watch = False):
    """
    This function trains a monkey with reinforcement learning. This function
    trains in batches, however, and runs on many simultanous game boards

    The DQN algorithm for each point in a batch.
    1) Get the policy's action.
    2) Get the consequent state (move the monkey).
    3) Get the immediate reward from the grid.
    4) Calculate the loss
        a) Calculate the quality of the move undertaken Q(s,a).
        b) Calculate the max_a Q(s',a) where s' is the consequent
           state of performing a from the state s.
        c) delta = Q(s,a) - r - gamma*max_a Q(s', a)
           where r is the immediate loss measured from the system.
        d) Loss is the Huber loss (smooth L1 loss) of delta.

    Args:
        brain: The brain that runs every monkey in every room.
        rooms: A list of rooms to use. Batch size is inferred from the length
            of this list.
        epochs: The number of epochs of training.
        gamma: The discount for the Bellman equation.
        epsilon: Default lambda function returning zero. A function that gives
            the value for epsilon based on epoch number.
        lr: The learning rate.
        watch: Default False. If True, will wait for the user to look at every
            iteration of the training.

    Returns:
        0: Training data in the form of list of tuples. First element is
        iteration number, second number is average loss over the
        iterations leading up to this report.
    """
    # First generate all the grids.
    grids = []
    for channel_map in rooms:
        # Create a new monkey
        new_monkey = monkey.Monkey(brain)

        # Find a good position for the monkey
        bad_position = True
        while bad_position:
            new_i = random.randrange(channel_map.size()[0])
            new_j = random.randrange(channel_map.size()[1])
            if all(channel_map[new_i, new_j, :] == torch.zeros(len(gl.BLOCK_TYPES))):
                bad_position = False
        new_monkey.pos = (new_i, new_j)

        # Create a new grid
        new_grid = grid.Grid([new_monkey], channel_map)

        # Add the grid to the list
        grids.append(new_grid)

    # Unpack epsilon if it exists
    epsilon_needed = False
    if g.monkeys[0].brain.pi == g.monkeys[0].brain.pi_epsilon_greedy:
        epsilon_needed = True

    # Generate the first run



def test_model(g, N, reset, loud=True):
    """
    This function will test the first monkey in the grid given for its
    score in the game after N resets of some number of turns each. The
    score is the sum of the food at the end of each reset.

    Args:
        g: The grid that the monkeys are on.
        N: The number of resets to do.
        reset: The number of turns per reset.
        loud: Whether to report progress or not.

    Returns:
        0: Average score over all resets.
    """
    # Set all monkeys to evaluation mode
    for monkey in g.monkeys:
        monkey.brain.eval()

    # Initialize score.
    score_record = []

    # Iterate over resets.
    for n in range(N):
        if (n+1)%50 == 0 and loud:
            print('Reset', n+1)

        # Randomize position
        invalid_spot = True
        while invalid_spot:
            i = random.randrange(g.width)
            j = random.randrange(g.height)
            # This spot can have a monkey placed on it
            if g.channel_map[gl.INDEX_BARRIER,i,j] == 0 and \
                g.channel_map[gl.INDEX_DANGER,i,j] == 0:
                # First teleport the monkey on the channel map\
                g.teleport_monkey(g.monkeys[0].pos, (i,j))
                # Update the position in the monkey object
                g.monkeys[0].pos = (i,j)
                # Flag for completion.
                invalid_spot = False

        # Set food value
        g.monkeys[0].food = reset+1

        # Revive monkey if dead
        g.monkeys[0].dead = False

        # Do turns
        for turn in range(reset):
            g.tick(0, invincible = True)
            # Check if dead
            if g.monkeys[0].dead:
                # No score contribution
                g.monkeys[0].food = 0
                break

        # Reset turn number
        g.turn_count = 0
        g.monkeys[0].age = 0

        # Add to score
        score_record.append(g.monkeys[0].food)

    score_record = torch.FloatTensor(score_record)
    average_score = score_record.mean().item()
    std = score_record.std().item()

    return average_score, std