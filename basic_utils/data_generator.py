import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from basic_utils.env_wrapper import Scaler
from models.agents import *


class Parallel_Path_Data_Generator:
    def __init__(self,
                 agent,
                 env,
                 n_worker,
                 path_num,
                 noise,
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
        self.agent = agent
        self.extra_info_name = noise.extra_info
        for i in range(self.n_worker):
            s = Queue()
            r = Queue()
            self.senders.append(s)
            self.receivers.append(r)
            self.workers.append(
                mp.Process(target=path_rollout, args=(agent, env, path_num, r, s, noise, render, action_repeat, i)))
        for worker in self.workers:
            worker.start()

    def __call__(self, num_episode=None):
        """
        Get a fixed number of paths.

        return:
            complete_paths: a list containing several dicts which contains the rewards,
                            actions, observations and not_dones
            path_info: rewards of each step in each path
            extra_info: extra information
        """
        episode_counter = 0
        paths = []
        noise_info = {k: 0 for k in self.extra_info_name}
        for i in range(self.n_worker):
            paths.append(defaultdict(list))

        while (num_episode is None) or episode_counter < num_episode:
            counter = 0
            complete_paths = []
            params = self.agent.get_params()
            for index in range(self.n_worker):
                if self.senders[index].empty():
                    self.senders[index].put((params, self.scaler.get()))

            while counter != self.n_worker:
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
                    for exk in self.extra_info_name:
                        extra_info[exk] = np.mean([k[exk][-1] for k in complete_paths])
                    yield complete_paths, path_info, extra_info


class Parallel_Memory_Data_Generator:
    def __init__(self,
                 agent,
                 memory,
                 env,
                 n_worker,
                 step_num,
                 noise,
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
               noise: a class to generate noise
               rand_explore_len: length for the agent to randomly select actions at the beginning of the game
               action_repeat: number of repeated actions
               render: whether display the game or not
        """
        mp.set_start_method('spawn')

        self.memory = memory
        self.n_worker = n_worker
        self.rand_explore_len = rand_explore_len
        self.extra_info_name = noise.extra_info
        self.agent = agent

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
                agent, env, step_num, noise, r, s, render, action_repeat, i)))
        for worker in self.workers:
            worker.start()

    def __call__(self):
        """
        Get a fixed number of paths.

        return:
            sample_return: a dict of randomly sampled samples which contains the rewards,
                            actions, observations and not_dones
            path_info: rewards of each step in each path
            extra_info: extra information
        """
        while True:
            counter = 0
            params = self.agent.get_params()
            for index in range(self.n_worker):
                if self.senders[index].empty():
                    self.senders[index].put((params, self.scaler.get()))

            while counter != self.n_worker:
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
                    extra_info = {"Memory_length": len(self.memory)}
                    for exk in self.extra_info_name:
                        extra_info[exk] = np.mean([k[exk][-1] for k in self.complete_paths])
                    self.complete_paths = []
                    yield sample_return, path_info, extra_info


class Memory_Data_Generator:
    def __init__(self,
                 agent,
                 memory,
                 env,
                 step_num,
                 noise,
                 rand_explore_len,
                 action_repeat,
                 render):
        """
           Generate several steps and save them to a replay memory.

           Args:
               agent: the agent for action selection
               env: the environment
               step_num: number of steps to generate at every call
               noise: a class to generate noise
               rand_explore_len: length for the agent to randomly select actions at the beginning of the game
               action_repeat: number of repeated actions
               render: whether display the game or not
        """

        self.memory = memory
        self.rand_explore_len = rand_explore_len
        self.extra_info_name = noise.extra_info
        self.agent = agent
        self.step_num = step_num
        self.env = env
        self.action_repeat = action_repeat
        self.render = render
        self.noise = noise

        self.complete_paths = []
        self.scaler = Scaler(env.observation_space_sca.shape)

    def __call__(self, num_episode=None, use_noise=True):
        """
        Get a fixed number of paths.

        return:
            sample_return: a dict of randomly sampled samples which contains the rewards,
                            actions, observations and not_dones
            path_info: rewards of each step in each path
            extra_info: extra information
        """
        data = defaultdict(list)
        count = 0
        episode_count = 0
        noise_info = {k: 0 for k in self.extra_info_name}
        while (num_episode is None) or episode_count < num_episode:
            self.env.set_scaler(self.scaler.get())
            ob, env_info = self.env.reset()
            self.noise.reset()
            now_repeat = 0
            for k in env_info:
                data[k].append(env_info[k])
            done = False
            while not done:
                if self.render:
                    self.env.render()
                data["observation"].append(ob)
                if now_repeat == 0:
                    action = self.agent.act(ob.reshape((1,) + ob.shape))[0]
                    if use_noise:
                        action, noise_info = self.noise.process_action(action)

                now_repeat = (now_repeat + 1) % self.action_repeat
                for k in noise_info:
                    data[k].append(noise_info[k])

                data["action"].append(action)
                ob, rew, done, env_info = self.env.step(action)
                data["next_observation"].append(ob)
                for k in env_info:
                    data[k].append(env_info[k])
                data["reward"].append(rew)
                data["not_done"].append(1 - done)

                count += 1

                self.memory.add((data['observation'][-1], data['action'][-1],
                                 data['next_observation'][-1], data['reward'][-1],
                                 data['not_done'][-1]))

                if done:
                    episode_count += 1
                    path = {k: np.array(data[k]) for k in data}
                    self.scaler.update(path['observation_raw'])
                    self.complete_paths.append(path)
                    data = defaultdict(list)

                if count >= self.step_num:
                    count = 0
                    sample_return = self.memory.sample() if len(self.memory) > self.rand_explore_len else None
                    extra_info = {"Memory_length": len(self.memory)}
                    if len(self.complete_paths) > 0:
                        path_info = [p['reward_raw'] for p in self.complete_paths]
                        for exk in self.extra_info_name:
                            extra_info[exk] = np.mean([k[exk][-1] for k in self.complete_paths])
                    else:
                        path_info = []
                    self.complete_paths = []
                    yield sample_return, path_info, extra_info


def path_rollout(agent,
                 env,
                 path_num,
                 require_q,
                 recv_q,
                 noise,
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
        noise.reset()
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
                action, noise_info = noise.process_action(action)
            now_repeat = (now_repeat + 1) % action_repeat
            for k in noise_info:
                single_data[k].append(noise_info[k])
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
                     noise,
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
        noise: a class to generate noise
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
        ob, env_info = env.reset()
        noise.reset()
        now_repeat = 0
        for k in env_info:
            single_data[k].append(env_info[k])
        done = False
        while not done:
            if render and process_id == 0:
                env.render()
            single_data["observation"].append(ob)
            if now_repeat == 0:
                action = agent.act(ob.reshape((1,) + ob.shape))[0]
                action, noise_info = noise.process_action(action)
            now_repeat = (now_repeat + 1) % action_repeat

            for k in noise_info:
                single_data[k].append(noise_info[k])
            single_data["action"].append(action)
            ob, rew, done, env_info = env.step(action)
            single_data["next_observation"].append(ob)
            for k in env_info:
                single_data[k].append(env_info[k])
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
