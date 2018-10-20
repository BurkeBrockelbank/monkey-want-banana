"""
Main program

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/main.py
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import math
import numpy as np

import global_variables as gl
import exceptions
import room_generator as rg
import brain
import monkey
import grid
import train

def plot_phase_2(loss_report, epsilon_guide_space, epsilon_explore_space, save_path = None, average_iters = 20):
    """
    This function plots the loss report from phase 2 type training (guided dqn)

    Args:
        loss_report: Formatted as
            [(n, loss, score, guide_epsilon, explore_epsilon), ...].
        save_path: Default None. If a string is given, it will try to save the plot
            to that location and not show it.
        average_iters: Number of iterations to average over for representing score trend.
    """
    n, loss, test_result, test_err, guide, explore = [np.array(x) for x in zip(*loss_report)]
    fig = plt.figure(figsize=(24, 6)) 
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2])
    fig = plt.figure()
    fig.suptitle('Learning averaged over %d iterations.' % (average_iters, ))

    ax1 = plt.subplot(gs[0])
    ax1.set_ylabel('Loss, Score')
    average_score = np.convolve(np.ones(average_iters)*(1/average_iters), test_result, mode='same')
    # score_err = ax1.fill_between(n, test_result-test_err, test_result+test_err, color='orange', alpha=0.4)
    # score_plot = ax1.plot(n, test_result, label='score', color='orange')
    average_score_plot = ax1.plot(n, average_score, color='brown', \
        label='Average score (%d iterations)' % (average_iters,))

    ax2 = plt.subplot(gs[1])
    ax2.set_ylabel('Loss')
    average_loss = np.convolve(np.ones(average_iters)*(1/average_iters), loss, mode='same')
    # loss_plot = ax1.plot(n, loss, label='loss', color='blue')
    average_loss_plot = ax2.plot(n, average_loss, color='blue', \
        label='Average loss (%d iterations)' % (average_iters,))

    ax3 = plt.subplot(gs[2])
    ax3.set_ylabel('explore ε')
    ax3.set_xlabel('Turn')
    ax3.set_yticks(epsilon_explore_space)
    explore_plot = ax3.plot(n, [x for x in explore], color='red', label='explore ε')

    ax4 = ax3.twinx()
    ax4.set_ylabel('guide ε')
    ax4.yaxis.tick_right()
    ax4.set_yticks(epsilon_guide_space)
    guide_plot = ax4.plot(n, guide, color='green', label='guide ε')

    plots = guide_plot + explore_plot
    ax3.legend(plots, [l.get_label() for l in plots], loc=0)

    plt.tight_layout()

    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)

def plot_phase_3(loss_report, save_path = None, average_iters = 100):
    """
    This function plots the loss report from phase 2 type training (guided dqn)

    Args:
        loss_report: Formatted as
            [(n, loss, score, guide_epsilon, explore_epsilon), ...].
        save_path: Default None. If a string is given, it will try to save the plot
            to that location and not show it.
        average_iters: Number of iterations to average over for representing score trend.
    """
    n, loss, test_result, test_err, guide, explore = [np.array(x) for x in zip(*loss_report)]
    fig = plt.figure(figsize=(24, 6)) 
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2])
    fig = plt.figure()
    fig.suptitle('Learning averaged over %d iterations.' % (average_iters, ))

    ax1 = plt.subplot(gs[0])
    ax1.set_ylabel('Score')
    average_score = np.convolve(np.ones(average_iters)*(1/average_iters), test_result, mode='same')
    # score_err = ax1.fill_between(n, test_result-test_err, test_result+test_err, color='orange', alpha=0.4)
    # score_plot = ax1.plot(n, test_result, label='score', color='orange')
    average_score_plot = ax1.plot(n, average_score, color='brown', \
        label='Average score (%d iterations)' % (average_iters,))

    ax2 = plt.subplot(gs[1])
    ax2.set_ylabel('Loss')
    average_loss = np.convolve(np.ones(average_iters)*(1/average_iters), loss, mode='same')
    # loss_plot = ax1.plot(n, loss, label='loss', color='blue')
    average_loss_plot = ax2.plot(n, average_loss, color='blue', \
        label='Average loss (%d iterations)' % (average_iters,))

    ax3 = plt.subplot(gs[2])
    ax3.set_ylabel('explore ε')
    ax3.set_xlabel('Turn')
    explore_plot = ax3.plot(n, [x for x in explore], color='red', label='explore ε')

    plt.tight_layout()

    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)



def dump_parameters(brain, path):
    out_F = open(path, 'w')
    out_F.write(str(list(brain.parameters())))
    out_F.close()

if __name__ == "__main__":
    # Some constants we will be using
    phase = 3
    gamma = 0.8
    lr_supervised = 0.001
    lr_reinforcement = 0.001
    epochs = 5
    batches = 10
    reports = 5
    N = 500
    epsilon_start = 0.00
    epsilon_end = 0.00
    n_epsilon = 20000
    epsilon_tuple = (epsilon_start, epsilon_end, n_epsilon)
    def epsilon(n):
        return (epsilon_start - epsilon_end)*\
            math.exp(-(n+1)/n_epsilon) + epsilon_end
    max_discount = 0.001
    CR_level = 2
    CR_block_index = 2

    # Create the map
    # room_start = rg.rand_room(500, [0.03,0,0.05,0.01])
    room_start = rg.png_to_channel('maps\\AdventureMapBananaLavaShrunk.png', [(0,0,0), (128,64,0), (255,242,0), (237,28,36)])
    # Create brain to train
    monkey_brain = brain.BrainV16()
    AI_brain = brain.BrainDecisionAI(gamma, gl.BANANA_FOOD-1, -1, gl.DEATH_REWARD, save_Q=True)
    # monkey_brain = brain.BrainDecisionAI(gamma, gl.BANANA_FOOD-1, -1, gl.DEATH_REWARD, save_Q=True) #########
    # Put brain in monkey in grid
    monkeys = [monkey.Monkey(monkey_brain)]
    test_monkeys = [monkey.Monkey(monkey_brain)]
    monkeys[0].pos = (len(room_start[1])//2,len(room_start[2])//2)
    test_monkeys[0].pos = (len(room_start[1])//2,len(room_start[2])//2)
    g = grid.Grid(monkeys, room_start)
    test_g = grid.Grid(test_monkeys, room_start.clone())

    # Make data paths for the monkey training data
    paths = ['data\\AdventureMapBananaLavaShrunk\\life'+str(i)+'.dat' for i in range(750)]

    # # Generate training data
    # train.monkey_training_data_1_life(50000, 100, 'data\\AdventureMapBananaLavaShrunkMediumBanana\\life', g, loud = [])
    # exit()
    

    # rg.play_record('data\\life300.dat')

    # train.clean_data(paths, [s.replace('.txt', 'CLEAN.txt') for s in paths])

    ######## PHASE 1 #############
    if phase == 1:
        print('PHASE 1: Copying')
        # Grid search
        train.grid_search_supervised(brain.BrainV17, 30, (100, 400, 10), (-4,-2,10), paths, \
        gamma, room_start, 'Phase_1_Data\\V17\\', score_tuple = (50,50))
        exit()

    # # Model testing
    # g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_greedy
    # print(train.test_model(g, 300, 50))
    # g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_epsilon_greedy
    # exit()

    # # Watch monkey train
    # monkey_brain.report = True
    # train.curated_bananas_dqn(g_CR, CR_level, 20, gamma, 0, 20, \
    #     block_index = CR_block_index, watch = True)
    # train.dqn_training(g, 60, gamma, 0, watch = True)
    # monkey_brain.report = False

    ####### PHASE 2 ########
    if phase == 2:
        print('PHASE 2: Detachment')
        folder = 'Phase_2_Data\\V16'
        guide_range = [0,0.1,0.2,0.35,0.5, 0.65,0.8,0.9,1]
        explore_range = [0,0.01,0.03,0.06]
        explore_override = [0.00, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
        monkey_brain.load_state_dict(torch.load('brainsave\\V16_batch190lr0.002154435496777296.brainsave'))
        total_training_data, percentage_list, epsilon_guide_history, epsilon_explore_history = \
            train.guide_search(g, test_g, gamma, lr_reinforcement, AI_brain, \
            guide_range, explore_range, \
            20, 10000, folder, initial_phase=10000, testing_tuple=(50,50), \
            override_explore = explore_override, number_of_tests = 20)
        plot_phase_2(total_training_data, guide_range, explore_range, \
            folder + '\\best_plot.png', average_iters = 20000)
        with open(folder + '\\report.txt', 'w') as out_f:
            out_f.write('Guide range:\n')
            out_f.write(str(guide_range)+'\n')
            out_f.write('\nExplore range:\n')
            out_f.write(str(explore_range)+'\n')
            out_f.write('\nPercentages:\n')
            out_f.write(str(percentage_list)+'\n')
            out_f.write('\nBest epsilon_guide:\n')
            out_f.write(str(epsilon_guide_history)+'\n')
            out_f.write('\nBest epsilon_explore:\n')
            out_f.write(str(epsilon_explore_history)+'\n')
            out_f.write('\nReport:\n')
            out_f.write(str(total_training_data))
        exit()


    ###### PHASE 3 ########
    if phase == 3:
        print('PHASE 3: Solitary Improvements')
        epsilon_func = lambda x: 0.04 * np.exp(-2 * x)
        points = 3000000
        filename = 'lr%d_e%d_%d_p%d' % (lr_reinforcement, 100*epsilon_func(0), 100*epsilon_func(1), points)
        folder = 'Phase_3_Data\\V17'
        monkey_brain.load_state_dict(torch.load('brainsave\\Phase_2_V17_p19g100e0.brainsave'))
        phase_3_data = train.unguided_dqn(g, test_g, points, gamma, lr_reinforcement, \
            testing_tuple = (50, 50), epsilon_explore = epsilon_func, \
            report_score = True, number_of_tests = 500)
        torch.save(monkey_brain.state_dict(), '%s\\%s.brainsave' % (folder, filename))
        plot_phase_3(phase_3_data, save_path = '%s\\%s.png' % (folder, filename), average_iters = 100000)
        with open('%s\\%s.txt' % (folder, filename), 'w') as out_f:
            out_f.write(str(phase_3_data))
        exit()

    if phase == 2.5:
        filename = 'Phase_3_V15'
        monkey_brain.load_state_dict(torch.load('brainsave\\Phase_2_V15.brainsave'))
        guide_range = [1]
        explore_range = [0,0.01,0.02,0.03,0.06,0.08,0.10,0.12]
        phase_3_data, percentage_list, epsilon_guide_history, epsilon_explore_history = \
            train.guide_search(g, test_g, gamma, lr_reinforcement, AI_brain, \
            guide_range, explore_range, \
            8, 15000, 'Phase_3_Data\\epsilon_search_V15', initial_phase=0, testing_tuple=(50,50))
        torch.save(monkey_brain.state_dict(), 'brainsave\\%s.brainsave' % (filename,))
        plot_phase_2(phase_3_data, guide_range, explore_range, \
            'Phase_3_Data\\epsilon_search_V15\\best_plot.png')
        with open('Phase_3_Data\\epsilon_search_V15\\report.txt', 'w') as out_f:
            out_f.write('Guide range:\n')
            out_f.write(str(guide_range)+'\n')
            out_f.write('\nExplore range:\n')
            out_f.write(str(explore_range)+'\n')
            out_f.write('\nPercentages:\n')
            out_f.write(str(percentage_list)+'\n')
            out_f.write('\nBest epsilon_guide:\n')
            out_f.write(str(epsilon_guide_history)+'\n')
            out_f.write('\nBest epsilon_explore:\n')
            out_f.write(str(epsilon_explore_history)+'\n')
            out_f.write('\nReport:\n')
            out_f.write(str(phase_3_data))
        exit()
        exit()
