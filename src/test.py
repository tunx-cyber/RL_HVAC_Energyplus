import matplotlib.pyplot as plt
from EnergyplusEnv import EnergyPlusEnvironment 
import actor_critic
import Config
import torch
cfg = Config.config()

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def train(cfg,envs):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    env = EnergyPlusEnvironment()
    env.seed(10)
    n_states  = envs.observation_space.shape[0]
    n_actions = envs.action_space.n
    model = actor_critic.actor_critic(n_states, n_actions, cfg)
    optimizer = cfg.optm
    step_idx    = 0
    test_rewards = []
    test_ma_rewards = []
    state = envs.reset()
    while step_idx < cfg.max_steps:
        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0
        # rollout trajectory
        for _ in range(cfg.n_steps):
            state = torch.FloatTensor(state).to(cfg.device)
            dist, value = model(state)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
            state = next_state
            step_idx += 1
            if step_idx % 200 == 0:
                test_reward = np.mean([test_env(env,model) for _ in range(10)])
                print(f"step_idx:{step_idx}, test_reward:{test_reward}")
                test_rewards.append(test_reward)
                if test_ma_rewards:
                    test_ma_rewards.append(0.9*test_ma_rewards[-1]+0.1*test_reward)
                else:
                    test_ma_rewards.append(test_reward) 
                # plot(step_idx, test_rewards)   
        next_state = torch.FloatTensor(next_state).to(cfg.device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)
        advantage = returns - values
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Finish training！')
    return test_rewards, test_ma_rewards