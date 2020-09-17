from runner import Runner
import argparse
import signal
from env_wrappers import *
from ic3net_envs.traffic_junction_env import TrafficJunctionEnv
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args




if __name__ == '__main__':
    for i in range(8):
        parser = get_common_args()
        args = parser.parse_args()

        if args.env_name == 'traffic_junction':
            env = TrafficJunctionEnv()
            if args.display:
                env.init_curses()
            env.init_args(parser)
        args = parser.parse_args()
        env.multi_agent_init(args)
        env = EnvWrapper(env)


        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
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
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
