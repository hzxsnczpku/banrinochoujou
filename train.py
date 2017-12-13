from models.agents import *
from basic_utils.data_generator import Path_Data_generator, Memory_Data_generator


class Path_Trainer:
    def __init__(self,
                 agent,
                 env,
                 n_worker=10,
                 path_num=10,
                 save_every=None,
                 render=False,
                 action_repeat=1,
                 print_every=10):
        self.callback = Callback()
        self.save_every = save_every
        self.print_every = print_every

        self.agent = agent
        self.env = env
        self.data_generator = Path_Data_generator(agent=self.agent,
                                                  env=self.env,
                                                  n_worker=n_worker,
                                                  path_num=path_num,
                                                  action_repeat=action_repeat,
                                                  render=render)

    def train(self):
        count = 1
        while True:
            self.data_generator.set_param(self.agent.get_params())

            paths, path_info, extra_info = self.data_generator.derive_data()

            u_stats, info = self.agent.update(paths)
            self.callback.add_update_info(u_stats)
            self.callback.add_path_info(path_info, extra_info)

            if self.callback.num_batches() >= self.print_every:
                count = self.callback.print_table()

            if self.save_every is not None and count % self.save_every == 0:
                self.agent.save_model('./save_model/' + self.env.name + '_' + self.agent.name)
                np.save('./save_score/' + self.env.name + '_' + self.agent.name, self.callback.scores)


class Mem_Trainer:
    def __init__(self,
                 agent,
                 memory,
                 env,
                 n_worker=1,
                 step_num=1,
                 explore_len=10000,
                 ini_epsilon=1.0,
                 final_epsilon=0.01,
                 rand_explore_len=1000,
                 action_repeat=1,
                 save_every=None,
                 render=False,
                 print_every=100):
        self.callback = Callback()
        self.save_every = save_every
        self.print_every = print_every

        self.agent = agent
        self.env = env
        self.memory = memory
        self.data_generator = Memory_Data_generator(agent=self.agent,
                                                    env=self.env,
                                                    memory=self.memory,
                                                    n_worker=n_worker,
                                                    rand_explore_len=rand_explore_len,
                                                    explore_len=explore_len,
                                                    ini_epsilon=ini_epsilon,
                                                    final_epsilon=final_epsilon,
                                                    step_num=step_num,
                                                    action_repeat=action_repeat,
                                                    render=render)

    def train(self):
        count = 1
        while True:
            self.data_generator.set_param(self.agent.get_params())

            paths, path_info, extra_info = self.data_generator.derive_data()

            u_stats, info = self.agent.update(paths)
            if info is not None:
                self.memory.update_priorities(info["idxes"], info["td_err"])
            self.callback.add_update_info(u_stats)
            self.callback.add_path_info(path_info, extra_info)

            if self.callback.num_batches() >= self.print_every:
                count = self.callback.print_table()

            if self.save_every is not None and count % self.save_every == 0:
                self.agent.save_model('./save_model/' + self.env.name + '_' + self.agent.name)
                np.save('./save_score/' + self.env.name + '_' + self.agent.name, self.callback.scores)
