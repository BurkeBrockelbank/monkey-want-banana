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
import random
import math

import global_variables as gl
import exceptions
import room_generator as rg
import brain
import monkey
import grid
import train

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
    room_start = rg.png_to_channel('img\\AdventureMapBananaLavaShrunkMoreBananas.png', [(0,0,0), (128,64,0), (255,242,0), (237,28,36)])
    # Create brain to train
    monkey_brain = brain.BrainV8()
    AI_brain = brain.BrainDecisionAI(gamma, 4, -1, gl.DEATH_REWARD, save_Q=True)
    # Put brain in monkey in grid
    monkeys = [monkey.Monkey(monkey_brain)]
    test_monkeys = [monkey.Monkey(monkey_brain)]
    monkeys[0].pos = (len(room_start[1])//2,len(room_start[2])//2)
    test_monkeys[0].pos = (len(room_start[1])//2,len(room_start[2])//2)
    g = grid.Grid(monkeys, room_start)
    test_g = grid.Grid(test_monkeys, room_start.clone())

    # # Make data paths for the monkey training data
    # paths = ['data\\life'+str(i)+'.dat' for i in range(600)]

    # # Generate training data
    # train.monkey_training_data_1_life(50000, 250, 'data\\life', g, loud = [])
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
    # train.grid_search_supervised(brain.BrainV8, 30, (245, 290, 10), (-3.5,-2.9,10), paths, \
    # gamma, max_discount, room_start, 'grid_search\\')

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
    # torch.save(monkey_brain.state_dict(), 'brainsave\\V4r_480-200-16-8-4_T0.brainsave')

    # Load brain from permanent memory
    monkey_brain.load_state_dict(torch.load('grid_search\\gamma0.8\\history_zoom\\batch332lr0.0007943279924802482.brainsave'))

    # monkey_brain.pi = monkey_brain.pi_greedy
    # for i in range(4000):
    #     g.tick(0, loud=[0], wait=True)

    # # # Model testing
    # g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_greedy
    # print(train.test_model(g, 100, 30))  
    # test_results = []
    # for r in range(5):
    #     test_results.append(train.test_model(g, 1000, 30))        
    #     print(test_results)
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

    # Reinforcment learning
    loss_report = train.guided_dqn(g, test_g, 20000, gamma, lr_reinforcement, \
    AI_brain)

    # Save the brain
    torch.save(monkey_brain.state_dict(), 'brainsave\\gridT0.brainsave')

    out_f = open('out.txt', 'a')
    out_f.write(str(loss_report))
    out_f.write('\n')
    out_f.close()
    plt.title('Learning ' + str(lr_reinforcement), )
    plt.xlabel('Turn')
    plt.ylabel('Loss')
    # plt.ylim(0,6)
    plt.plot(*zip(*loss_report))
    plt.show()

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
