import torch
import os
from network.base_net import RNN
from network.commnet import CommNet
from network.g2anet import G2ANet


class Reinforce:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape  
        # 根据参数决定RNN的输入维度
        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents
        self.args = args

        # 每个agent选动作的网络,输出当前agent所有动作对应的概率
        # 用该概率选动作的时候还需要用softmax再运算一次。
        if self.args.alg == 'reinforce+g2anet':
            print('Init alg reinforce+g2anet')
            self.eval_rnn = G2ANet(actor_input_shape, args)
        else:
            raise Exception("No such algorithm")

        if self.args.cuda:
            self.eval_rnn.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.env_name + '/'
        # 如果存在模型则加载模型
        if self.args.load != '':
            if os.path.exists(self.model_dir + self.args.load + '_rnn_params.pkl'):
                path_rnn = self.model_dir + self.args.load + '_rnn_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                print('Successfully load the model: {}'.format(path_rnn))
            else:
                raise Exception("No model!")

        self.rnn_parameters = list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr=args.lr_actor)
        self.args = args

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden
        self.eval_hidden = None

    def learn(self, batch, max_episode_len, train_step, epsilon):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, mini_mask, alive_mask = batch['u'], batch['r'], batch['complete'], batch['alive_mask']
        if self.args.cuda:
            r = r.cuda()
            u = u.cuda()
            mini_mask = mini_mask.cuda()
            alive_mask = alive_mask.cuda()

        # 得到每条经验的return, (episode_num, max_episode_len， n_agents)
        n_return = self._get_returns(r, mini_mask, alive_mask, max_episode_len)

        # 每个agent的所有动作的概率 (episode_num, max_episode_len， n_agents，n_actions)
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)

        # 每个agent的选择的动作对应的概率 (episode_num, max_episode_len， n_agents)
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)
        log_pi_taken = torch.log(pi_taken)

        # loss函数，(episode_num, max_episode_len, n_agents)
        loss = - (n_return * log_pi_taken).sum() / (episode_num * max_episode_len * self.n_agents)
        self.rnn_optimizer.zero_grad()
        loss.backward()
        if self.args.alg == 'reinforce+g2anet':
            torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()
        # print('Actor loss is', loss)

    def _get_returns(self, r, mini_mask, alive_mask, max_episode_len):
        n_return = torch.zeros_like(r)
        n_return[:, max_episode_len-1] = r[:, max_episode_len-1]
        for transition_idx in range(max_episode_len - 2, -1, -1):
            n_return[:, transition_idx] = (r[:, transition_idx] + self.args.gamma * n_return[:, transition_idx + 1]) * mini_mask[:,transition_idx]
        # print('n_return', n_return[0])
        # print('mini_mask', mini_mask[0])
        # print('r', r[0])
        # print('alive_mask', alive_mask[0])
        # print((n_return*alive_mask)[0])
        return n_return * alive_mask

    def _get_actor_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)
        # 给inputs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            # 因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把inputs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        episode_num = batch['o'].shape[0]
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
            # inputs维度为(episode_num * n_agents,inputs_shape)，得到的outputs维度为(episode_num * n_agents, n_actions)
            outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            # 把q_eval维度重新变回(episode_num, n_agents, n_actions)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)
        # 得的action_prob是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        action_prob = torch.stack(action_prob, dim=1).cpu()
        action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / self.args.n_actions)

        if self.args.cuda:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')