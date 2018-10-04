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

def plot_phase_2(loss_report, epsilon_guide_space, epsilon_explore_space, save_path = None):
    """
    This function plots the loss report from phase 2 type training (guided dqn)

    Args:
        loss_report: Formatted as
            [(n, loss, score, guide_epsilon, explore_epsilon), ...].
        save_path: Default None. If a string is given, it will try to save the plot
            to that location and not show it.
    """
    n, loss, test_result, test_err, guide, explore = [np.array(x) for x in zip(*loss_report)]
    fig = plt.figure(figsize=(12, 6)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    fig = plt.figure()
    plt.title('Learning ' + str(lr_reinforcement))

    ax1 = plt.subplot(gs[0])
    ax1.set_ylabel('Loss, Score')
    average_score = np.convolve(np.ones(10)*0.1, test_result, mode='same')
    score_err = ax1.fill_between(n, test_result-test_err, test_result+test_err, color='orange', alpha=0.4)
    score_plot = ax1.plot(n, test_result, label='score', color='orange')
    average_score_plot = ax1.plot(n, average_score, label='10-point average score', color='brown')
    loss_plot = ax1.plot(n, loss, label='loss', color='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('guide ε')
    ax2.yaxis.tick_right()
    ax2.set_yticks(epsilon_guide_space)
    guide_plot = ax2.plot(n, guide, color='green', label='guide ε')

    ax3 = plt.subplot(gs[1])
    ax3.set_ylabel('explore ε')
    ax3.set_xlabel('Turn')
    ax3.set_yticks(epsilon_explore_space)
    explore_plot = ax3.plot(n, [x for x in explore], color='red', label='explore ε')

    plot = loss_plot + score_plot + guide_plot + explore_plot
    labels = [l.get_label() for l in plot]
    ax1.legend(plot, labels, loc=0)

    plt.tight_layout()

    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)

# def plot_phase_2(loss_report, save_path = None):
#     """
#     This function plots the loss report from phase 2 type training (guided dqn)

#     Args:
#         loss_report: Formatted as
#             [(n, loss, score, guide_epsilon, explore_epsilon), ...].
#         save_path: Default None. If a string is given, it will try to save the plot
#             to that location and not show it.
#     """
#     n, loss, test_result, test_err, guide, explore = [np.array(x) for x in zip(*loss_report)]

#     fig = plt.figure()
#     plt.title('Learning ' + str(lr_reinforcement))

#     ax1 = fig.add_subplot(111)
#     ax1.set_xlabel('Turn')
#     ax1.set_ylabel('Loss, Score')
#     score_err = ax1.fill_between(n, test_result-test_err, test_result+test_err, color='orange', alpha=0.4)
#     score_plot = ax1.plot(n, test_result, label='score', color='orange')
#     loss_plot = ax1.plot(n, loss, label='loss', color='blue')

#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Epsilon')
#     ax2.yaxis.tick_right()
#     guide_plot = ax2.plot(n, guide, color='green', label='guide ε')
#     explore_plot = ax2.plot(n, [x*10 for x in explore], color='red', label='explore ε × 10')

#     plot = loss_plot + score_plot + guide_plot + explore_plot
#     labels = [l.get_label() for l in plot]
#     ax1.legend(plot, labels, loc=0)

#     if save_path == None:
#         plt.show()
#     else:
#         plt.savefig(save_path)

def dump_parameters(brain, path):
    out_F = open(path, 'w')
    out_F.write(str(list(brain.parameters())))
    out_F.close()

if __name__ == "__main__":
    # Some constants we will be using
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
    monkey_brain = brain.BrainV15()
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
    # train.monkey_training_data_1_life(50000, 100, 'data\\AdventureMapBananaLavaShrunk\\life', g, loud = [])
    # exit()
    

    # rg.play_record('data\\life300.dat')

    # train.clean_data(paths, [s.replace('.txt', 'CLEAN.txt') for s in paths])

    # # Load brain from permanent memory
    # monkey_brain.load_state_dict(torch.load('brainsave\\V4_480-200-16-8-4_T0.brainsave'))

    # # Train the monkey
    # loss_report = train.Q_supervised_training(2, 6, paths, \
    #     monkey_brain, lr_supervised)

    # # Train the monkey
    # loss_report = train.cross_entropy_supervised_training(8, 6, paths, \
    #     monkey_brain, lr_supervised)

    # # Grid search
    # train.grid_search_supervised(brain.BrainV15, 30, (100, 520, 20), (-4,-2,20), paths, \
    # gamma, room_start, 'grid_search\\V15\\')
    # exit()

    # # Train the monkey
    # loss_report = train.supervised_columns(100, 300, paths, monkey_brain, \
    #     gamma, max_discount, lr_supervised, 10, \
    #     intermediate='brainsave\\intermediate.brainsave')

    # out_f = open('out.txt', 'a')
    # out_f.write(str(loss_report))
    # out_f.write('\n')
    # out_f.close()
    # plt.title('Supervised Training lr' + str(lr_supervised) + \
    #     ' ' + str(paths)[6:22])
    # plt.xlabel('Epoch')
    # plt.ylabel('Average Loss per Data Point')
    # plt.plot(*zip(*loss_report))
    # plt.show()

    # # Save the brain
    # torch.save(monkey_brain.state_dict(), 'brainsave\\V4r_480-200-16-8-4_T0_no_truncate.brainsave')

    # # Load brain from permanent memory
    # monkey_brain.load_state_dict(torch.load('grid_search\\gamma0.8\\history_zoom\\batch332lr0.0007943279924802482.brainsave'))
    # Load brain from permanent memory
    # monkey_brain.load_state_dict(torch.load('brainsave\\batch226lr0.0029763521160930395.brainsave'))
    # # Load brain from permanent memory
    # monkey_brain.load_state_dict(torch.load('brainsave\\total_training.brainsave'))

    # monkey_brain.pi = monkey_brain.pi_greedy
    # for i in range(4000):
    #     g.tick(0, loud=[0], wait=True)

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

    # # Curated learning 
    # loss_report = train.curated_bananas_dqn(g_CR, CR_level, 4000, gamma, \
    #     lr_reinforcement, 20, block_index = CR_block_index, \
    #     random_start = False, epsilon = lambda x: epsilon(x))

    # # Reinforcment learning
    # loss_report = train.dqn_training(g, 50000, gamma, lr_reinforcement, \
    # epsilon = epsilon, watch = False)

    # # # Reinforcment learning
    # epsilon_guide = train.epsilon_interpolation([0,10,20,40,80,100],[0,0,0.2,0.3,0.8,1])
    # epsilon_explore = train.epsilon_interpolation([0,40,80,100],[0,0,0.02,0])
    # loss_report = train.guided_dqn(g, test_g, 100000, gamma, lr_reinforcement, \
    # AI_brain, epsilon_guide, epsilon_explore)

    # Guided DQN search
    guide_range = [0,0.1,0.2,0.35,0.5, 0.65,0.8,0.9,1]
    explore_range = [0,0.01,0.03,0.06]
    explore_override = [0,0,0.01,0.02,0.03,0.04,0.04,0.04,0.03,0.02,0.01,0]
    monkey_brain.load_state_dict(torch.load('brainsave\\batch499lr0.0004281332076061517.brainsave'))
    total_training_data, percentage_list, epsilon_guide_history, epsilon_explore_history = \
        train.guide_search(g, test_g, gamma, lr_reinforcement, AI_brain, \
        guide_range, explore_range, \
        12, 10000, 'guide_search\\guide_search_V15', initial_phase=20000, testing_tuple=(300,50), \
        override_explore = explore_override)
    plot_phase_2(total_training_data, guide_range, explore_range, \
        'guide_search\\guide_search_V15\\best_plot.png')
    out_f = open('guide_search\\guide_search_V15\\report.txt', 'w')
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

    # # Save the brain
    # torch.save(monkey_brain.state_dict(), 'brainsave\\total_training.brainsave')

    # out_f = open('out.txt', 'a')
    # out_f.write(str(loss_report))
    # out_f.write('\n')
    # out_f.close()

    # plot_phase_2(loss_report)

    # # Model testing
    # g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_greedy
    # print(train.test_model(g, 100, 30))  
    # test_results = []
    # for r in range(5):
    #     test_results.append(train.test_model(g, 1000, 30))        
    #     print(test_results)
    # g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_epsilon_greedy
    # exit()

    # train.curated_bananas_dqn(g_CR, CR_level, 20, gamma, 0, 20, \
    #     block_index = CR_block_index, watch = True)
    # train.dqn_training(g, 60, gamma, 0, watch = True)


    # # Build monkeys
    # monkeys = [monkey.Monkey(brain.BrainLinearAI()) for i in \
    #     range(50)]
    # for monkey in monkeys:
    #     # Place the monkeys a bit away from the walls (10 blocks).
    #     i = random.randrange(10,gl.RAND_ROOM_WIDTH-10)
    #     j = random.randrange(10,gl.RAND_ROOM_WIDTH-10)
    #     monkey.pos = (i,j)
    # g = grid.Grid(monkeys, room_start)


    # # Generate training data from the A.I.
    # train.monkey_training_data(1000000, paths, g, loud=[])


    # # Load brain from permanent memory
    # monkey_brain.load_state_dict(torch.load('brainsave_v2.txt'))

    # # Supervised monkey training
    # dump_parameters(monkey_brain, 'brain0.txt')
    # loss_data = train.supervised_training(epochs, ['data_channels.txt'], \
    #     monkey_brain, gamma, lr, reports)
    # dump_parameters(monkey_brain, 'brain1.txt')

    # # Reinforcement monkey training
    # for i in range(100):
    #     total_reward = train.dqn_training(g, N, gamma, lr, \
    #         epsilon_data = (epsilon_start, epsilon_end, n_epsilon), \
    #         watch = False)
    #     print(i,total_reward)

    # # Watch monkey train
    # total_reward = train.dqn_training(g, 20, gamma, lr_reinforcement, \
    #     epsilon_data = (epsilon_start, epsilon_end, n_epsilon), \
    #     watch = True)

    # for i in range(300):

    #     total_reward = train.dqn_training(g, N, gamma, lr_reinforcement, \
    #         epsilon_data = (epsilon_start, epsilon_end, n_epsilon), \
    #         watch = False)

    #     # Save the training data
    #     out_file = open('reward_report.txt', 'a')
    #     out_file.write(str(loss_data))
    #     out_file.write('\n')
    #     out_file.close()
    #     # Load the old reports
    #     to_show = train.loadRecords('reward_report.txt')
    #     # Grab the most recent epoch and separate it (to colour it differently)
    #     older_data = to_show[:-reports]
    #     newer_data = to_show[-reports:]
    #     # Plot the report record
    #     plt.title('Supervised Portion of DQN Transfer Learning')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Quality Loss (Sum of Squares)')
    #     plt.plot(*zip(*older_data), color='blue')
    #     plt.plot(*zip(*newer_data), color='red')
    #     plt.savefig('./img/reinforcement.png')
    #     plt.clf()

    #     # Test monkey
    #     g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_greedy
    #     print(train.test_model(g, 100, 30))
    #     g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_epsilon_greedy
