# Replay buffer

from common import *

# Running mean & stddev
class Running:
    def __init__(self, capacity):
        self.data = np.zeros([capacity], dtype = np.float32)
        self.size = 0
        self.n    = 0 # n is technically different from size here
        self.ptr  = 0
        self.cap  = capacity

        self.avg = 0
        self.m2   = 0

    def mean(self):
        return self.avg

    def sdev(self):
        return np.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0

    def add(self, x):
        goingaway = self.data[self.ptr]

        if self.ptr != self.size:
            self._minus(goingaway)

        self.n += 1
        assert self.n <= self.cap, f"Running stats overflow: ptr {self.ptr}, size {self.size}, n {self.n}"

        oldmean   = self.avg
        self.avg += (x - oldmean) / self.n
        self.m2  += (x - oldmean) * (x - self.avg)

        self.data[self.ptr] = x
        self.size           = min(self.size + 1, self.cap)
        self.ptr            = (self.ptr + 1) % self.cap

    def _minus(self, x):
        oldmean   = self.avg
        self.n   -= 1
        self.avg -= (x - oldmean) / self.n
        self.m2  -= (x - self.avg) * (x - oldmean)

class ReplayBuffer:
    def __init__(self, capacity, action_space, device):
        self.dev = device

        self.obs      = [None] * capacity
        self.next_obs = [None] * capacity

        # Split discrete and continuous actions
        self.acts_d = np.zeros([capacity, len(action_space["d"].nvec)], dtype=np.int32)
        self.acts_c = np.zeros([capacity, action_space["c"].shape[0]], dtype=np.float32)

        self.rewards  = np.zeros([capacity], dtype = np.float32)
        self.lasts    = np.zeros(capacity, dtype = np.float32) # actually not last

        self.stats    = Running(capacity)

        # Statistics for normalization
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0
        self.count = 0

        self.capacity = capacity
        self.ptr      = 0
        self.size     = 0


    def __len__(self):
        return self.size


    def add(self, obs, act, next_obs, reward, last):
        self.obs[self.ptr]      = obs
        self.acts_c[self.ptr]   = act["c"]
        self.acts_d[self.ptr]   = act["d"]
        self.next_obs[self.ptr] = next_obs
        self.lasts[self.ptr]    = last

        self.stats.add(reward)
        mean, sdev = self.stats.mean(), self.stats.sdev()
        if sdev == 0:
            self.rewards[self.ptr] = 0
        else:
            self.rewards[self.ptr] = (reward - mean) / sdev

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, number):
        idxs = np.random.choice(self.size, size = number, replace = False)
        return dict(
            obs      = [self.obs[i] for i in idxs],
            next_obs = [self.next_obs[i] for i in idxs],
            acts     = {
                "c": torch.FloatTensor(self.acts_c[idxs]).to(self.dev),
                "d": torch.LongTensor(self.acts_d[idxs]).to(self.dev),
            },
            rewards  = torch.FloatTensor(self.rewards[idxs]).reshape(-1, 1).to(self.dev),
            lasts    = torch.BoolTensor(self.lasts[idxs]).reshape(-1, 1).to(self.dev)
        )