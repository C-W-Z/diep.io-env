import gymnasium
import numpy as np

from common import PNetwork, torch
from feat_encoder import *

class ActingAgent(object):
    def __init__(self, dim_obs, action_space, path):
        self.dev = "cpu"

        self.te = EncoderWrapper(self.dev, TANK_FEAT_META).to(self.dev)
        self.pe = EncoderWrapper(self.dev, POLY_FEAT_META).to(self.dev)
        self.be = EncoderWrapper(self.dev, BULLET_FEAT_META).to(self.dev)

        self.policy = PNetwork(dim_obs, action_space, self.te, self.pe, self.be, self.dev)

        self.load(path)

    def act(self, obs):
        def t2np(action):
            detached = {}
            for k, v in action.items():
                if isinstance(v, torch.Tensor):
                    detached[k] = v[0].detach().cpu().numpy()
                else:
                    detached[k] = v
            return detached

        _, _, act = self.policy.sample(obs)
        return t2np(act)

    def load(self, path):
        save = torch.load(path)
        self.policy.load_state_dict(save['policy_state_dict'])
        self.te.load_state_dict(save['te_state_dict'])
        self.pe.load_state_dict(save['pe_state_dict'])
        self.be.load_state_dict(save['be_state_dict'])
        self.policy.eval()
        self.te.eval()
        self.pe.eval()
        self.be.eval()
