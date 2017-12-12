import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from basic_utils.env_wrapper import Scaler
from models.agents import *


class Path_Data_generator:
    def __init__(self, agent, env, n_worker, path_num, render):
        """
        A data generator which generates a fixed number of paths at every call.

        param:
        agent_generator: a function which returns the agent
        env_generator: a function which returns the environment
        n_worker: number of parallel agents
        path_num: number of paths generated by each worker
        render: whether to display the game or not
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
                mp.Process(target=path_rollout, args=(agent, env, path_num, r, s, render, i)))
        for worker in self.workers:
            worker.start()

    def set_param(self, params):
        """
        Set the parameters of the sub agents.

        param:
        params: a dict containing the parameters of the model
        """
        for index in range(self.n_worker):
            if self.senders[index].empty():
                self.senders[index].put((params, self.scaler.get()))

    def derive_data(self):
        """
        Get a fixed number of paths

        return:
        complete_paths: a list containing several dicts which contains the rewards,
         actions, observations and not_dones
        path_info: information about each path
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
    def __init__(self, cfg):
        mp.set_start_method('spawn')
        self.cfg = cfg
        if 'Prioritized' not in cfg:
            self.memory = ReplayBuffer(self.cfg)
        else:
            self.memory = PrioritizedReplayBuffer(self.cfg)

        self.workers = []
        self.senders = []
        self.receivers = []
        self.paths = []
        for i in range(self.cfg["n_worker"]):
            self.paths.append(defaultdict(list))
        self.complete_paths = []
        self.scaler = Scaler(self.cfg["observation_space"].shape)
        for i in range(self.cfg["n_worker"]):
            s = Queue()
            r = Queue()
            self.senders.append(s)
            self.receivers.append(r)
            self.workers.append(mp.Process(target=mem_path_rollout, args=(self.cfg, r, s, i)))
        for worker in self.workers:
            worker.start()

    def set_param(self, params):
        for index in range(self.cfg["n_worker"]):
            if self.senders[index].empty():
                self.senders[index].put((params, self.scaler.get()))

    def set_info(self, info):
        if info is not None:
            self.memory.update_priorities(info["idxes"], info["td_err"])

    def derive_data(self):
        for i in range(self.cfg["n_worker"]):
            while not self.receivers[i].empty():
                single_trans = self.receivers[i].get()
                self.memory.add((single_trans['observation'][0], single_trans['action'][0],
                                 single_trans['next_observation'][0], single_trans['reward'][0],
                                 single_trans['not_done'][0]))
                derived_data = {'reward_raw': single_trans['reward_raw'], 'not_done': single_trans['not_done'],
                                'observation_raw': single_trans['observation_raw'], 'epsilon': single_trans['epsilon']}
                done = merge_dict(self.paths[i], derived_data)

                if done:
                    path = {k: np.array(self.paths[i][k]) for k in self.paths[i]}
                    self.scaler.update(path['observation_raw'])
                    self.complete_paths.append(path)
                    self.paths[i] = defaultdict(list)

        sample_return = self.memory.sample() if len(self.memory) > self.cfg["rand_explore_len"] else None
        path_info = [p['reward_raw'] for p in self.complete_paths]
        extra_info = {"Memory_length": len(self.memory),
                      'Epsilon': np.mean([k['epsilon'][-1] for k in self.complete_paths])}
        self.complete_paths = []
        return sample_return, path_info, extra_info


class ES_path_Data_generator:
    def __init__(self, cfg):
        mp.set_start_method('spawn')
        self.cfg = cfg
        self.workers = []
        self.senders = []
        self.receivers = []
        self.scaler = Scaler(self.cfg["observation_space"].shape)
        for i in range(2 * self.cfg["n_kid"]):
            s = Queue()
            r = Queue()
            self.senders.append(s)
            self.receivers.append(r)
            self.workers.append(mp.Process(target=path_rollout, args=(self.cfg, r, s, i)))
        for worker in self.workers:
            worker.start()

    def set_param(self, params):
        self.noise_seed = np.random.randint(0, 2 ** 32 - 1, size=self.cfg['n_kid'], dtype=np.uint32).repeat(2)
        for index in range(2 * self.cfg["n_kid"]):
            seed = self.noise_seed[index]
            np.random.seed(seed)
            change = turn_into_cuda(
                torch.from_numpy(sign(index) * self.cfg['sigma'] * np.random.randn(params.numel()))).float()
            new_params = params + change
            self.senders[index].put((new_params, self.scaler.get()))

    def set_info(self, info):
        pass

    def derive_data(self):
        paths = []
        complete_paths = []
        for i in range(2 * self.cfg["n_kid"]):
            paths.append(defaultdict(list))
        counter = 0
        while True:
            for i in range(2 * self.cfg["n_kid"]):
                while not self.receivers[i].empty():
                    single_trans = self.receivers[i].get()
                    if single_trans is None:
                        counter += 1
                    else:
                        done = merge_dict(paths[i], single_trans)
                        if done:
                            path = {k: np.array(paths[i][k]) for k in paths[i]}
                            path['index'] = i
                            path['seed'] = self.noise_seed
                            self.scaler.update(path['observation_raw'])
                            del path['observation_raw']
                            complete_paths.append(path)
                            paths[i] = defaultdict(list)
            if counter == 2 * self.cfg["n_kid"]:
                path_info = [p['reward_raw'] for p in complete_paths]
                extra_info = {}
                return complete_paths, path_info, extra_info


def path_rollout(agent, env, path_num, require_q, recv_q, render, process_id=0):
    """
    Generates several paths for each worker.

    param:
    agent_generator: a function which returns the agent
    env_generator: a function which returns the environment
    path_num: number of paths to generate
    require_q: a queue to put the generated data
    recv_q: a queue to get the params
    render: whether to display the game or not
    param process_id: id of the worker
    """
    params, scale = recv_q.get()
    agent.set_params(params)

    single_data = defaultdict(list)
    count = 0
    while True:
        env.set_scaler(scale)
        ob, info = env.reset()
        for k in info:
            single_data[k].append(info[k])
        done = False
        while not done:
            if render and process_id == 0:
                env.render()
            single_data["observation"].append(ob)
            action = agent.act(ob.reshape((1,) + ob.shape))
            single_data["action"].append(action[0])
            ob, rew, done, info = env.step(agent.process_act(action[0]))
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


def step_rollout(agent, env, step_num, require_q, recv_q, render, process_id=0):
    """
        Generates several paths for each worker.

        param:
        agent_generator: a function which returns the agent
        env_generator: a function which returns the environment
        path_num: number of paths to generate
        require_q: a queue to put the generated data
        recv_q: a queue to get the params
        render: whether to display the game or not
        param process_id: id of the worker
        """
    params, scale = recv_q.get()
    agent.set_params(params)

    single_data = defaultdict(list)
    count = 0
    while True:
        env.set_scaler(scale)
        ob, info = env.reset()
        for k in info:
            single_data[k].append(info[k])
        done = False
        while not done:
            if render and process_id == 0:
                env.render()
            single_data["observation"].append(ob)
            action = agent.act(ob.reshape((1,) + ob.shape))
            single_data["action"].append(action[0])
            ob, rew, done, info = env.step(agent.process_act(action[0]))
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

def mem_path_rollout(cfg, require_q, recv_q, process_id=0):
    agent = get_agent(cfg)
    params, scale = recv_q.get()
    agent.set_params(params)

    env = Env_wrapper(cfg)
    single_data = defaultdict(list)
    while True:
        env.set_scaler(scale)
        ob, info = env.reset()
        for k in info:
            single_data[k].append(info[k])
        done = False
        while not done:
            if cfg['render'] and process_id == 0:
                env.render()
            single_data["observation"].append(ob)
            action = agent.act(ob.reshape((1,) + ob.shape))
            single_data["action"].append(action)
            ob, rew, done, info = env.step(action)
            for k in info:
                single_data[k].append(info[k])
            single_data["reward"].append(rew)
            single_data["not_done"].append(1 - done)
            single_data["next_observation"].append(ob)
            single_data['epsilon'].append(agent.epsilon)

            require_q.put(single_data)

            single_data = defaultdict(list)
            params, scale = recv_q.get()
            agent.set_params(params)

