"""
Feature encoder:
    Translates variable-length observations into fixed-size representations.
    The output of this can be fed to your networks as per usual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

STATS_FEAT_META = [
    0, 0, # positions
    0, # r
    0, 0, # velocity
    0, # hp
    45, # level
    34, # skill point avail
    8, 8, 8, 8, 8, 8, 8, 8 # skills
]

POLY_FEAT_META = [
    0, 0, # pos
    0, # r
    0, 0, # velocity
    0, # hp
    2, 2, 2 # sides
]

TANK_FEAT_META = [
    0, 0, # pos
    0, # r
    0, 0, # velocity
    0, # hp
    45 # level
]

BULLET_FEAT_META = [
    0, 0, # pos
    0, # r
    0, 0, # velocity
    2, # friendly or not
]

"""
Encodes one of the sets (tank/polygon/whatever)
"""
class DeepSetEncoder(nn.Module):
    """
    This one uses deep sets.
    """
    def __init__(self,
        # lists marking the discrete variables with
        # values > 0 being their number of possible values
        feat_meta,
        # network settings
        embed_dim = 8, hidden_dim = 32, output_dim = 64
    ):
        super().__init__()

        self.feat_meta = feat_meta
        self.embed_dim = embed_dim

        self.embeddings = nn.ModuleList([
            nn.Embedding(n, embed_dim) if n > 0 else None for n in feat_meta
        ])

        # dimension after embeddings
        inter_dim = sum([
            embed_dim if n > 0 else 1 for n in feat_meta
        ])

        # Encode individual thing
        self.phi = nn.Sequential(
            nn.Linear(inter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Aggregate
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        No Batching
        """
        N, F = x.shape

        if N == 0:  # no objects in this category
            return torch.zeros(F).to(x.device)

        features = []

        for i, meta in enumerate(self.feature_meta):
            col = x[:, i]
            if meta > 0:
                # Discrete: embed
                col = col.long()
                emb = self.embeddings[i](col)
                features.append(emb)
            else:
                # Continuous: use as-is with 1D shape
                features.append(col.unsqueeze(-1))

        # Concatenate all features
        x_cat = torch.cat(features, dim=-1)  # (N, total_input_dim)

        # Apply phi to each element
        phi_out = self.phi(x_cat)  # (N, hidden_dim)

        # Aggregate
        pooled = phi_out.sum(dim=1)  # (hidden_dim)

        # Get final output
        out = self.rho(pooled)  # (output_dim)

        return out

"""
Batching wrapper.
"""
class EncoderWrapper(nn.Module):
    def __init__(self, feat_meta, embed_dim, hidden_dim, output_dim, method="deepset"):
        super().__init__()

        if method == "deepset":
            self.encoder = DeepSetEncoder(feat_meta, embed_dim, hidden_dim, output_dim)
        else:
            raise NotImplementedError()

    def forward(self, x):
        """
        x: list of B observations (may be different length)
        """
        results = [self.encoder(o) for o in x]
        return torch.stack(results, dim=0)