import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from basic_utils.env_wrapper import Scaler
from models.agents import *
from gym.spaces import Discrete, Box


class Path_Data_generator:
    def __init__(self,
                 agent,
                 env,
                 n_worker,
                 path_num,
                 action_repeat,
                 render):
        """
        Generate several whole paths.

        Args:
            agent: the agent for action selection
            env: the environment
            n_worker: the number of parallel workers
            path_num: number of paths to return at every call
            action_repeat: number of repeated actions
            render: whether display the game or not
        """
        mp.set_start_method('spawn')
        self.n_worker = n_worker
        self.workers = []
        self.senders = []
        self.receivers = []
        self.scaler = Scaler(env.observation_space_sca.shape)
        for i in range(self.n_worker):
            s = Queue()
            r = Queue()
            self.senders.append(s)
            self.receivers.append(r)
            self.workers.append(
                mp.Process(target=path_rollout, args=(agent, env, path_num, r, s, render, action_repeat, i)))
        for worker in self.workers:
            worker.start()

    def set_param(self, params):
        """
        Set the parameters for the sub agents.

        Args:
            params: a dict containing the parameters of the model
        """
        for index in range(self.n_worker):
            if self.senders[index].empty():
                self.senders[index].put((params, self.scaler.get()))

    def derive_data(self):
        """
        Get a fixed number of paths.

        return:
            complete_paths: a list containing several dicts which contains the rewards,
                            actions, observations and not_dones
            path_info: rewards of each step in each path
            extra_info: extra information
        """
        paths = []
        complete_paths = []
        for i in range(self.n_worker):
            paths.append(defaultdict(list))
        counter = 0
        while True:
            for i in range(self.n_worker):
                while not self.receivers[i].empty():
                    single_trans = self.receivers[i].get()
                    if single_trans is None:
                        counter += 1
                    else:
                        done = merge_dict(paths[i], single_trans)
                        if done:
                            path = {k: np.array(paths[i][k]) for k in paths[i]}
                            self.scaler.update(path['observation_raw'])
                            del path['observation_raw']
                            complete_paths.append(path)
                            paths[i] = defaultdict(list)
            if counter == self.n_worker:
                path_info = [p['reward_raw'] for p in complete_paths]
                extra_info = {}
                return complete_paths, path_info, extra_info


class Memory_Data_generator:
    def __init__(self,
                 agent,
                 memory,
                 env,
                 n_worker,
                 step_num,
                 ini_epsilon,
                 final_epsilon,
                 explore_len,
                 rand_explore_len,
                 action_repeat,
                 render):
        """
           Generate several steps and save them to a replay memory.

           Args:
               agent: the agent for action selection
               env: the environment
               n_worker: the number of parallel workers
               step_num: number of steps to generate at every call
               ini_epsilon: initial epsilon
               final_epsilon: final epsilon
               explore_len: length for epsilon to decay
               rand_explore_len: length for the agent to randomly select actions at the beginning of the game
               action_repeat: number of repeated actions
               render: whether display the game or not
        """
        mp.set_start_method('spawn')

        self.memory = memory
        self.n_worker = n_worker
        self.rand_explore_len = rand_explore_len

        self.workers = []
        self.senders = []
        self.receivers = []
        self.paths = []
        for i in range(n_worker):
            self.paths.append(defaultdict(list))
        self.complete_paths = []
        self.scaler = Scaler(env.observation_space_sca.shape)
        for i in range(n_worker):
            s = Queue()
            r = Queue()
            self.senders.append(s)
            self.receivers.append(r)
            self.workers.append(mp.Process(target=mem_step_rollout, args=(
                agent, env, step_num, ini_epsilon, final_epsilon, explore_len, r, s, render, action_repeat, i)))
        for worker in self.workers:
            worker.start()

    def set_param(self, params):
        """
        Set the parameters for the sub agents.

        Args:
            params: a dict containing the parameters of the model
        """
        for index in range(self.n_worker):
            if self.senders[index].empty():
                self.senders[index].put((params, self.scaler.get()))

    def derive_data(self):
        """
        Get a fixed number of paths.

        return:
            sample_return: a dict of randomly sampled samples which contains the rewards,
                            actions, observations and not_dones
            path_info: rewards of each step in each path
            extra_info: extra information
        """
        counter = 0
        while True:
            for i in range(self.n_worker):
                while not self.receivers[i].empty():
                    single_trans = self.receivers[i].get()
                    if single_trans is None:
                        counter += 1
                    else:
                        self.memory.add((single_trans['observation'][0], single_trans['action'][0],
                                         single_trans['next_observation'][0], single_trans['reward'][0],
                                         single_trans['not_done'][0]))
                        derived_data = {'reward_raw': single_trans['reward_raw'], 'not_done': single_trans['not_done'],
                                        'observation_raw': single_trans['observation_raw'],
                                        'epsilon': single_trans['epsilon']}
                        done = merge_dict(self.paths[i], derived_data)

                        if done:
                            path = {k: np.array(self.paths[i][k]) for k in self.paths[i]}
                            self.scaler.update(path['observation_raw'])
                            self.complete_paths.append(path)
                            self.paths[i] = defaultdict(list)
            if counter == self.n_worker:
                sample_return = self.memory.sample() if len(self.memory) > self.rand_explore_len else None
                path_info = [p['reward_raw'] for p in self.complete_paths]
                extra_info = {"Memory_length": len(self.memory),
                              'Epsilon': np.mean([k['epsilon'][-1] for k in self.complete_paths])}
                self.complete_paths = []
                return sample_return, path_info, extra_info


def path_rollout(agent,
                 env,
                 path_num,
                 require_q,
                 recv_q,
                 render,
                 action_repeat,
                 process_id=0):
    """
    Generates several paths for each worker.

    Args:
        agent: the agent for action selection
        env: the environment
        path_num: number of paths to return at every call
        require_q: a queue to put the generated data
        recv_q: a queue to get the params
        render: whether display the game or not
        action_repeat: number of repeated actions
        process_id: id of the worker
    """
    params, scale = recv_q.get()
    agent.set_params(params)

    single_data = defaultdict(list)
    count = 0
    while True:
        env.set_scaler(scale)
        ob, info = env.reset()
        agent.reset()
        now_repeat = 0
        for k in info:
            single_data[k].append(info[k])
        done = False
        while not done:
            if render and process_id == 0:
                env.render()
            single_data["observation"].append(ob)
            if now_repeat == 0:
                action = agent.act(ob.reshape((1,) + ob.shape))[0]
            now_repeat = (now_repeat + 1) % action_repeat
            single_data["action"].append(action)
            ob, rew, done, info = env.step(agent.process_act(action))
            single_data["next_observation"].append(ob)
            for k in info:
                single_data[k].append(info[k])
            single_data["reward"].append(rew)
            single_data["not_done"].append(1 - done)

            require_q.put(single_data)
            single_data = defaultdict(list)
        count += 1
        if count >= path_num:
            require_q.put(None)
            params, scale = recv_q.get()
            agent.set_params(params)
            count = 0


def mem_step_rollout(agent,
                     env,
                     step_num,
                     ini_epsilon,
                     final_epsilon,
                     explore_len,
                     require_q,
                     recv_q,
                     render,
                     action_repeat,
                     process_id=0):
    """
    Generates several paths for each worker.

    Args:
        agent: the agent for action selection
        env: the environment
        step_num: number of steps to generate at every call
        ini_epsilon: initial epsilon
        final_epsilon: final epsilon
        explore_len: length for epsilon to decay
        require_q: a queue to put the generated data
        recv_q: a queue to get the params
        render: whether display the game or not
        action_repeat: number of repeated actions
        process_id: id of the worker
    """

    params, scale = recv_q.get()
    agent.set_params(params)
    epsilon = ini_epsilon
    if isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
    epsilon_decay = (epsilon - final_epsilon) / explore_len

    single_data = defaultdict(list)
    count = 0
    while True:
        env.set_scaler(scale)
        ob, info = env.reset()
        agent.reset()
        now_repeat = 0
        for k in info:
            single_data[k].append(info[k])
        done = False
        while not done:
            if render and process_id == 0:
                env.render()
            single_data["observation"].append(ob)
            if now_repeat == 0:
                action = agent.act(ob.reshape((1,) + ob.shape))[0]
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, action_dim)
                if epsilon > final_epsilon:
                    epsilon -= epsilon_decay
            now_repeat = (now_repeat + 1) % action_repeat

            single_data['epsilon'].append(epsilon)
            single_data["action"].append(action)
            ob, rew, done, info = env.step(action)
            single_data["next_observation"].append(ob)
            for k in info:
                single_data[k].append(info[k])
            single_data["reward"].append(rew)
            single_data["not_done"].append(1 - done)

            require_q.put(single_data)
            single_data = defaultdict(list)
            count += 1
        if count >= step_num:
            require_q.put(None)
            params, scale = recv_q.get()
            agent.set_params(params)
            count = 0
