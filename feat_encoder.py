"""
Feature encoder:
    Translates variable-length observations into fixed-size representations.
    The output of this can be fed to your networks as per usual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSetEncoder(nn.Module):
    """
    This one uses deep sets.
    """
    def __init__(self,
        # dimensions of each part of the obs
        stats_dim, tank_feats_dim, polygon_feats_dim, bullet_feats_dim,
        # lists marking the discrete variables with
        # values > 0 being their number of possible values
        stats_discr, tank_discr, polygon_discr, bullet_discr,
        # network settings
        embed_dim = 8, hidden_dim = 32, output_dim = 64
    ):
        super().__init__()

