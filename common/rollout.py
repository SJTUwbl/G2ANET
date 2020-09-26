import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time

# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    def generate_episode(self, epoch, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay
            self.env.close()
        o, u, r, u_onehot, complete, alive_mask = [], [], [], [], [], []
        obs = self.env.reset(epoch)
        terminated = False
        step = 0
        episode_reward = 0
        self.agents.policy.init_hidden(1)
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            actions, actions_onehot = [], []

            # get the weights of all actions for all agents
            weights = self.agents.get_action_weights(np.array(obs))

            # choose action for each agent
            for agent_id in range(self.n_agents):
                action = self.agents.choose_action(weights[agent_id], epsilon, evaluate)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action.numpy())
                actions_onehot.append(action_onehot)
            actions = np.array(actions)

            next_obs, reward, terminated, info = self.env.step(actions)
            o.append(obs)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            r.append(reward)
            complete.append(1 - info['is_completed'])
            alive_mask.append(info['alive_mask'])
            episode_reward += reward
            obs = next_obs
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # last obs
        o.append(obs)
        o_next = o[1:]
        o = o[:-1]

        episode = dict(o=o.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       o_next=o_next.copy(),
                       u_onehot=u_onehot.copy(),
                       complete=complete.copy(),
                       alive_mask=alive_mask.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        success = self.env.get_stat()['success']
        add_rate = self.env.get_stat()['add_rate']
        return episode, success, add_rate
