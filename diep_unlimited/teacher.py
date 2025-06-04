# Training code

from common import *
from sakuya import Agent
from env import DiepIOEnvBasic

# environment
env = DiepIOEnvBasic()

# agent
agent0 = Agent(
    16 + 64 * 3, env.action_space,
    lr_pol = 1e-4, lr_cri = 2e-4, lr_alp = 1e-4, lr_enc = 1e-4,
    alpha = 0.2, gamma = 0.98, tau = 0.005, eta = 0.1, beta = 0.2,
    replay_buf_size = 100000, batch_size = 4096,
    load = False, auto_alpha = True
)
agent1 = Agent(
    16 + 64 * 3, env.action_space,
    lr_pol = 1e-4, lr_cri = 2e-4, lr_alp = 1e-4, lr_enc = 1e-4,
    alpha = 0.2, gamma = 0.98, tau = 0.005, eta = 0.1, beta = 0.2,
    replay_buf_size = 100000, batch_size = 4096,
    load = False, auto_alpha = True
)


# numbers
WARMUP      = 10000
TOTAL_STEPS = 1000000
SAVE_PER_EP = 5

n_steps   = 0
for episode in itertools.count(1):
    start = time.time()

    actor_losses  = [0]
    critic_losses = [0]

    ep_reward = 0
    done   = False
    obs, _ = env.reset()

    while not done:
        # sample action
        if n_steps < WARMUP:
            action0 = env.action_space.sample()
            action1 = env.action_space.sample()
        else:
            action0 = agent0.act(obs["agent_0"])
            action1 = agent1.act(obs["agent_1"])

        # step
        next_obs, rewards, dones, _, _ = env.step({
            "agent_0": action0,
            "agent_1": action1,
        })
        ep_reward += rewards["agent_0"]

        # remember
        agent0.remember(obs["agent_0"], action0, next_obs["agent_0"], rewards["agent_0"], not dones["agent_0"])
        agent1.remember(obs["agent_1"], action1, next_obs["agent_1"], rewards["agent_1"], not dones["agent_1"])

        # update
        if n_steps >= WARMUP and len(agent0.mem) >= agent0.batch_size:
            critic_l, actor_l = agent0.update()
            agent1.update() # synched

            critic_losses.append(critic_l)
            actor_losses.append(actor_l)

        done = dones["__all__"]
        obs = next_obs
        n_steps += 1

    end = time.time()
    elapsed = end - start

    print(f"[{elapsed:.0f}s]\tEpisode\t{episode}\tReward\t{ep_reward:.0f}\tAlpha\t{agent0.alpha.item():.4f}\tC\t{np.mean(critic_losses):.02f}\tA\t{np.mean(actor_losses):.02f}")


    if episode % SAVE_PER_EP == 0:
        agent0.save("diep_agent0", episode)
        agent1.save("diep_agent1", episode)

    if n_steps > TOTAL_STEPS:
        break