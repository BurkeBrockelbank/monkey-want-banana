"""
This is the loss module. It includes loss functions.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/loss.py
"""

import torch

import global_variables as gl


def L1ClampLoss(Q, Q_pi, a):
    """
    This calculates the loss as all the sum of the nonnegative values of
    f(Q-Q_pi, a) where f(x, i)_j = x_i if i != j and |x_i| if i = j.
    """
    Q_pi_vec = Q_pi.repeat(len(gl.WASD))
    delta = torch.add(Q, torch.neg(Q_pi_vec))

    flipper = torch.eye(len(Q), dtype=torch.float)
    if delta[a] < 0:
        flipper[a,a] = -1
    delta = torch.mv(flipper, delta)

    delta = torch.clamp(delta, min=0.0)

    loss = torch.sum(delta)
    return loss
