# Hopefully generic Soft Actor-Critic that will just work across all three tasks

from common import *
from replay import ReplayBuffer
from feat_encoder import EncoderWrapper, TANK_FEAT_META, POLY_FEAT_META, BULLET_FEAT_META

class Agent:
    def __init__(self, dim_obs, action_space,
        lr_cri = 1e-4, lr_pol = 1e-4, lr_alp = 1e-4, lr_enc = 1e-4,
        alpha = 0.2, gamma = 0.9173, tau = 0.005, eta = 0.1, beta = 0.2,
        replay_buf_size = 1000000, batch_size = 384,
        load = False, auto_alpha = True
    ):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Input\t{dim_obs}\tOutput\t{action_space}")

        self.alpha = torch.tensor(alpha, requires_grad = True, device = self.dev)
        self.gamma = gamma
        self.tau   = tau
        self.eta   = eta
        self.beta  = beta

        self.mem        = ReplayBuffer(replay_buf_size, action_space, self.dev)
        self.batch_size = batch_size

        self.te = EncoderWrapper(self.dev, TANK_FEAT_META).to(self.dev)
        self.pe = EncoderWrapper(self.dev, POLY_FEAT_META).to(self.dev)
        self.be = EncoderWrapper(self.dev, BULLET_FEAT_META).to(self.dev)
        tep = list(self.te.parameters())
        pep = list(self.pe.parameters())
        bep = list(self.be.parameters())
        self.ep = tep + pep + bep
        ep_id = {id(p) for p in self.ep}

        self.policy = PNetwork(dim_obs, action_space, self.te, self.pe, self.be, self.dev).to(self.dev)
        self.critic = QNetwork(dim_obs, action_space, self.te, self.pe, self.be, self.dev).to(self.dev)
        self.pp = [p for p in self.policy.parameters() if id(p) not in ep_id]
        self.cp = [p for p in self.critic.parameters() if id(p) not in ep_id]

        self.target = QNetwork(dim_obs, action_space, self.te, self.pe, self.be, self.dev).to(self.dev)
        self.target.load_state_dict(self.critic.state_dict())
        self.target.eval()

        self.pol_op = optim.AdamW(
            self.pp,
            lr = lr_pol,
            betas = (0.9, 0.9)
        )
        self.cri_op = optim.AdamW(
            self.cp,
            lr = lr_cri,
            betas = (0.9, 0.9)
        )
        self.enc_op = optim.AdamW(
            self.ep,
            lr = lr_enc,
            betas = (0.9, 0.9)
        )

        # entropy optimization
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.entropy_target = -6
            self.log_alpha      = torch.tensor(np.log(alpha), requires_grad = True, device = self.dev)
            self.alpha_op       = optim.AdamW([self.log_alpha], lr = lr_alp)

        if load:
            self.load("saves/.bin")


    def act(self, obs):
        def t2np(action):
            detached = {}
            for k, v in action.items():
                if isinstance(v, torch.Tensor):
                    detached[k] = v[0].detach().cpu().numpy()
                else:
                    detached[k] = v
            return detached

        # training mode
        act, _, _ = self.policy.sample(obs)
        return t2np(act)

    def remember(self, obs, act, next_obs, reward, last):
        self.mem.add(obs, act, next_obs, reward, last)

    def update(self):
        # replay
        samples = self.mem.sample(self.batch_size)

        states      = samples["obs"]      # These two are np arrays
        next_states = samples["next_obs"] # the networks will tensorfy them fine
        actions     = samples["acts"]
        rewards     = samples["rewards"]
        lasts       = samples["lasts"]

        # calculate critic
        with torch.no_grad():
            next_acts, next_log_probs, _ = self.policy.sample(next_states)
            next_q_tgt1, next_q_tgt2     = self.target(next_states, next_acts)

            min_next_q_tgt = torch.min(next_q_tgt1, next_q_tgt2) - self.alpha * next_log_probs
            next_q         = rewards + self.gamma * min_next_q_tgt * lasts

        q1, q2           = self.critic(states, actions)
        q1_loss, q2_loss = F.mse_loss(q1, next_q), F.mse_loss(q2, next_q)
        total_loss       = q1_loss + q2_loss

        # optimize critic
        self.cri_op.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.cri_op.step()

        # calculate policy
        pi, log_pi, _ = self.policy.sample(states)
        q1_pi, q2_pi  = self.critic(states, pi)
        min_q_pi      = torch.min(q1_pi, q2_pi)

        policy_loss   = torch.mean((self.alpha * log_pi) - min_q_pi)

        # optimize policy
        self.pol_op.zero_grad()
        policy_loss.backward()
        self.pol_op.step()

        # optimize entropy
        if self.auto_alpha:
            alpha_loss = -1 * torch.mean(self.log_alpha * (log_pi + self.entropy_target).detach())
            self.alpha_op.zero_grad()
            alpha_loss.backward()
            self.alpha_op.step()
            self.alpha = torch.exp(self.log_alpha)

        # soft update for continuity
        self._sync_target()

        return total_loss.detach().item(), policy_loss.detach().item()


    def save(self, name, number):
        if not os.path.exists('saves/'):
            os.makedirs('saves/')

        path = f"saves/save_{name}_{number}.bin"
        print(f"Saving to {path}")

        torch.save(
            {
                'policy_state_dict': self.policy.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'target_state_dict': self.target.state_dict(),
                'te_state_dict': self.te.state_dict(),
                'pe_state_dict': self.pe.state_dict(),
                'be_state_dict': self.be.state_dict(),
                'policy_optimizer_state_dict': self.pol_op.state_dict(),
                'critic_optimizer_state_dict': self.cri_op.state_dict(),
                'encoder_optimizer_state_dict': self.enc_op.state_dict(),
            },
            path
        )

    def load(self, path, evaluate = False):
        print(f"Loading from {path}")

        save = torch.load(path)
        self.policy.load_state_dict(save['policy_state_dict'])
        self.critic.load_state_dict(save['critic_state_dict'])
        self.target.load_state_dict(save['target_state_dict'])
        self.te.load_state_dict(save['te_state_dict'])
        self.pe.load_state_dict(save['pe_state_dict'])
        self.be.load_state_dict(save['be_state_dict'])
        self.cri_op.load_state_dict(save['critic_optimizer_state_dict'])
        self.pol_op.load_state_dict(save['policy_optimizer_state_dict'])
        self.enc_op.load_state_dict(save['encoder_optimizer_state_dict'])

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.target.train()


    def _sync_target(self):
        for target_param, param in zip(self.target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)