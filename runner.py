import numpy as np
import os
import sys
import time
from common.rollout import CommRolloutWorker
from agent.agent import CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.plt_success = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.env_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        train_steps = 0
        for epoch in range(self.args.n_epoch):
            epoch_success = 0
            add_rate = 0
            epoch_begin_time = time.time()
            for n in range(self.args.epoch_size):
                episodes = []
                batch_success = 0
                # one batch, 收集self.args.n_episodes个episodes
                for episode_idx in range(self.args.n_episodes):
                    episode, success, add_rate = self.rolloutWorker.generate_episode(epoch, episode_idx)
                    episodes.append(episode)
                    batch_success += success
                # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
                episode_batch = episodes[0]
                episodes.pop(0)
                for episode in episodes:
                    for key in episode_batch.keys():
                        episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

                if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                    self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                    train_steps += 1
                else:
                    self.buffer.store_episode(episode_batch)
                    for train_step in range(self.args.train_steps):
                        mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                        self.agents.train(mini_batch, train_steps)
                        train_steps += 1
                print('\t\t batch success: {:.3f}'.format(batch_success / self.args.n_episodes))
                epoch_success += batch_success
            
            print('Run {}, train epoch {}'.format(num, epoch))
            epoch_time = time.time() - epoch_begin_time
            print('Time {:.2f}s'.format(epoch_time))
            print('Add_rate: {:.2f}\t Success: {:.2f}'
                .format(add_rate, epoch_success / self.args.epoch_size / self.args.n_episodes))
            self.plt_success.append(epoch_success / self.args.epoch_size / self.args.n_episodes)
        print('random seed', self.args.seed)
        self.plt(num)

    def evaluate(self):
        print('yes')
        epoch_success = 0
        for epoch in range(self.args.evaluate_epoch):
            _, success, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            epoch_success += success
        return epoch_success / self.args.evaluate_epoch


    def plt(self, num):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 1])
        plt.plot(range(len(self.plt_success)), self.plt_success)
        plt.xlabel('epoch*{}'.format(self.args.n_epoch))
        plt.ylabel('success rate')
        plt.savefig(self.save_path + '/plt_{}.png'.format(self.args.seed), format='png')
        plt.show()

        np.save(self.save_path + '/success_rate_{}'.format(self.args.seed), self.plt_success)
