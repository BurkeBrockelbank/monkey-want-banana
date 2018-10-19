"""
This is the loss module. It includes loss functions.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/loss.py
"""

import torch

import functools

import global_variables as gl


def L1_clamp_loss(Q, Q_pi, a):
    """
    This calculates the loss as all the sum of the nonnegative values of
    f(Q-Q_pi, a) where f(x, i)_j = x_i if i != j and |x_i| if i = j.
    """
    Q_pi_vec = Q_pi.view((-1,1)).repeat(1,len(gl.WASD))
    delta = torch.add(Q, torch.neg(Q_pi_vec))

    flipper = torch.ones(delta.size(), dtype=torch.float)
    for i, a_val in enumerate(a):
        if delta[i,a_val] < 0:
            flipper[i,a_val] = -1

    delta = torch.mul(flipper, delta)

    delta = torch.clamp(delta, min=0.0)

    loss = torch.sum(delta, dim=1)
    return loss

def L1_no_clamp(Q, Q_pi, a):
    """
    This calculates the loss as just the L1 loss between Q[a] and Q_pi.
    """
    loss = torch.abs(Q[range(len(Q)), a]-Q_pi)
    return loss

def leaky_L1_clamp_loss(leak, Q, Q_pi, a):
    """
    This calculates the loss as all the sum of the nonnegative values of
    f(Q-Q_pi, a) where f(x, i)_j = x_i if i != j and |x_i| if i = j.
    """
    Q_pi_vec = Q_pi.view((-1,1)).repeat(1,len(gl.WASD))
    delta = torch.add(Q, torch.neg(Q_pi_vec))

    flipper = torch.ones(delta.size(), dtype=torch.float)
    for i, a_val in enumerate(a):
        if delta[i,a_val] < 0:
            flipper[i,a_val] = -1

    flipper = torch.ones(delta.size(), dtype=torch.float)
    for i, a_val in enumerate(a):
        for j in range(delta.size()[1]):
            if delta[i, j] < 0:
                if j == a_val:
                    flipper[i, j] = -1
                else:
                    flipper[i, j] = -leak

    delta = torch.mul(flipper, delta)

    loss = torch.sum(delta, dim=1)
    return loss

def leaky_L1_clamp_loss_generator(leak=0.1):
    return functools.partial(leaky_L1_clamp_loss, leak)