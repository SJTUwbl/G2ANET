from runner import Runner
import argparse
import signal
import sys
from env_wrappers import *
from ic3net_envs.traffic_junction_env import TrafficJunctionEnv
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
import random
import torch
import numpy


def set_seed(args):
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

if __name__ == '__main__':
    for i in range(1):
        parser = get_common_args()
        args = parser.parse_args()

        if args.env_name == 'traffic_junction':
            env = TrafficJunctionEnv()
            if args.display:
                env.init_curses()
            env.init_args(parser)
        args = parser.parse_args()
        print('learn:', args.learn)
        if args.seed == -1:
            args.seed = random.randint(0,10000)
        set_seed(args)

        env.multi_agent_init(args)
        env = EnvWrapper(env)

        args = get_reinforce_args(args)
        args = get_g2anet_args(args)

        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        print(args)

        def signal_handler(signal, frame):
            print('You pressed Ctrl+C! Exiting gracefully.')
            if args.display:
                env.end_display()
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        runner = Runner(env, args)
        if args.learn:
            print('no')
            runner.run(i)
        else:
            win_rate = runner.evaluate()
            print('The win rate of {} is  {:.2f}'.format(args.alg, win_rate))
            break
        env.close()

