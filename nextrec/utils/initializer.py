"""
Initialization utilities for NextRec

Date: create on 13/11/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import torch.nn as nn


def get_initializer(init_type='normal', activation='linear', param=None):
    """
    Get parameter initialization function.
        
    Examples:
        >>> init_fn = get_initializer('xavier_uniform', 'relu')
        >>> init_fn(tensor)
        >>> init_fn = get_initializer('normal', param={'mean': 0.0, 'std': 0.01})
    """
    param = param or {}

    try:
        gain = param.get('gain', nn.init.calculate_gain(activation, param.get('param', None)))
    except ValueError:
        gain = 1.0  # for custom activations like 'dice'
    
    def initializer_fn(tensor):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(tensor, gain=gain)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(tensor, gain=gain)
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(tensor, a=param.get('a', 0), nonlinearity=activation)
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(tensor, a=param.get('a', 0), nonlinearity=activation)
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(tensor, gain=gain)
        elif init_type == 'normal':
            nn.init.normal_(tensor, mean=param.get('mean', 0.0), std=param.get('std', 0.0001))
        elif init_type == 'uniform':
            nn.init.uniform_(tensor, a=param.get('a', -0.05), b=param.get('b', 0.05))
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        return tensor

    return initializer_fn
