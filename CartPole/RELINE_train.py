# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************
# ***                          Using Reinforcement Learning for Load Testing of Video Games                          ***
# ***                                                 Game: CartPole                                                 ***
# ***                                RELINE: Cross Entropy Method + info injected bugs                               ***
# ***                                           Training for 200 iterations                                          ***
# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************


import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

HIDDEN_SIZE = 128  # neural network size
BATCH_SIZE = 16    # num episodes
PERCENTILE = 70    # elite episodes
MAX_ITER = 1000     # training iterations

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    # OBSERVATION:
    # - x coordinate of the stick's center of mass
    # - speed
    # - angle to the platform
    # - angular speed
    sm = nn.Softmax(dim=1)
    flag_injected_bug_spotted = [False, False]
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        if -0.35 < next_obs[0] < -0.3 and not flag_injected_bug_spotted[0]:
            reward += 50
            flag_injected_bug_spotted[0] = True
        if 0.3 < next_obs[0] < 0.35 and not flag_injected_bug_spotted[1]:
            reward += 50
            flag_injected_bug_spotted[1] = True
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            flag_injected_bug_spotted = [False, False]
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

def evaluate_model_with_bugs(env, net, episodes=50):
    net.eval()
    total_reward = 0.0
    sm = nn.Softmax(dim=1)

    for _ in range(episodes):
        obs = env.reset()
        done = False
        flag_injected_bug_spotted = [False, False]
        episode_reward = 0.0

        while not done:
            obs_v = torch.FloatTensor([obs])
            act_probs_v = sm(net(obs_v))
            act_probs = act_probs_v.data.numpy()[0]
            action = np.random.choice(len(act_probs), p=act_probs)
            next_obs, reward, done, _ = env.step(action)

            # 버그 체크
            if -0.35 < next_obs[0] < -0.3 and not flag_injected_bug_spotted[0]:
                reward += 50
                flag_injected_bug_spotted[0] = True
            if 0.3 < next_obs[0] < 0.35 and not flag_injected_bug_spotted[1]:
                reward += 50
                flag_injected_bug_spotted[1] = True

            episode_reward += reward
            obs = next_obs

        total_reward += episode_reward

    return total_reward / episodes

# **********************************************************************************************************************
# *                                                   TRAINING START                                                   *
# **********************************************************************************************************************


if __name__ == "__main__":
    print('\n\n****************************************************************')
    print("* RL-baseline model's training on CartPole game is starting... *")
    print('****************************************************************\n')
    env = gym.make("CartPole-v0")
    env._max_episode_steps = 1000  # episode length
    log_dir = "runs/cartpole_baseline2"

    writer = SummaryWriter(log_dir=log_dir)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    old_net = Net(obs_size, HIDDEN_SIZE, n_actions)
    try:
        old_net.load_state_dict(torch.load('./best_model_RELINE'))
        old_reward_mean = evaluate_model_with_bugs(env, old_net)
        print(f"기존 model_RELINE의 평균 보상: {old_reward_mean:.1f}")
    except FileNotFoundError:
        print("기존 model_RELINE가 없음, 기준 없이 새 모델 저장")
        old_reward_mean = -float('inf')

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if iter_no == MAX_ITER:
            print("Training ends")
            writer.close()
            print("\    /\ \n )  ( ')  meow!\n(  /  )\n \(__)|")
            torch.save(net.state_dict(), './model_RELINE')

            final_reward_mean = evaluate_model_with_bugs(env, net)
            print(final_reward_mean)

            if final_reward_mean > old_reward_mean:
                torch.save(net.state_dict(), './best_model_RELINE')
                print(" 기존 best_model_RELINE보다 성능이 좋아서 업데이트됨!")
            else:
                print(" 기존 best_model_RELINE가 더 우수하여 업데이트하지 않음.")
            
            break

#                                                                                                               \    /\
#                                                                                                                )  ( ')
#                                                                                                               (  /  )
#                                                                                                                \(__)|
