"""
This is the loss module. It includes loss functions.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss

class L1ClampLoss(torch.nn.modules.loss._Loss):
    """
    Creates a criterion that evaluates L1 on one of the elements of a quality
    and the ReLU ramp function on the rest of the qualities

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.L1ClampLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(L1ClampLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, Q, Q_pi, a):
        """
        Args:
            Q: A vector of qualities
            Q_pi: The actual quality of the move
            a: The action that was undertaken

        Returns:
            0: The loss
        """
        delta = Q - Q_pi
        if delta[a] < 0:
            delta[a] = -delta[a]
        return F.relu(delta)