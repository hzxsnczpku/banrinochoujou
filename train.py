from models.agents import *
from basic_utils.data_generator import Parallel_Path_Data_Generator
from basic_utils.env_wrapper import Scaler


class Path_Trainer:
    def __init__(self,
                 agent,
                 env,
                 data_generator,
                 data_processor,
                 save_every=None,
                 print_every=10,
                 log_dir_name=None):
        self.callback = Callback(log_dir_name)
        self.save_every = save_every
        self.print_every = print_every

        self.agent = agent
        self.env = env
        self.data_generator = data_generator
        self.data_processor = data_processor

    def train(self):
        count = 1
        for paths, path_info, extra_info in self.data_generator():
            processed_path = self.data_processor(paths)
            u_stats, info = self.agent.update(processed_path)
            self.callback.add_update_info(u_stats)
            self.callback.add_path_info(path_info, extra_info)

            if self.callback.num_batches() >= self.print_every:
                count = self.callback.print_table()

            if self.save_every is not None and count % self.save_every == 0:
                self.agent.save_model('./save_model/' + self.env.name + '_' + self.agent.name)


class ES_Trainer:
    def __init__(self,
                 agent,
                 env,
                 path_num=10,
                 n_worker=10,
                 save_every=None,
                 render=False,
                 action_repeat=1,
                 print_every=10):
        self.callback = Callback()
        self.save_every = save_every
        self.print_every = print_every

        self.agent = agent
        self.env = env
        self.data_generator = Parallel_Path_Data_Generator(agent=self.agent,
                                                           env=self.env,
                                                           n_worker=n_worker,
                                                           path_num=path_num,
                                                           action_repeat=action_repeat,
                                                           render=render)

    def train(self):
        count = 1
        while True:
            index = 0
            total_paths = []
            for new_params in self.agent.get_params():
                self.data_generator.set_param(new_params)
                paths, path_info, extra_info = self.data_generator.derive_data()
                for path in paths:
                    path['index'] = index
                total_paths += paths
                index += 1
                self.callback.add_path_info(path_info, extra_info)

            u_stats, info = self.agent.update(total_paths)
            self.callback.add_update_info(u_stats)

            if self.callback.num_batches() >= self.print_every:
                count = self.callback.print_table()

            if self.save_every is not None and count % self.save_every == 0:
                self.agent.save_model('./save_model/' + self.env.name + '_' + self.agent.name)


class Memory_Trainer:
    def __init__(self,
                 agent,
                 memory,
                 env,
                 data_generator,
                 data_processor,
                 eval_every=50,
                 save_every=None,
                 print_every=100,
                 log_dir_name=None):
        self.callback = Callback(log_dir_name)
        self.save_every = save_every
        self.print_every = print_every
        self.eval_every = eval_every

        self.agent = agent
        self.env = env
        self.memory = memory
        self.data_generator = data_generator
        self.data_processor = data_processor

    def train(self):
        count = 1
        while True:
            for paths, path_info, extra_info in self.data_generator(self.eval_every):
                if paths is not None:
                    processed_path = self.data_processor([paths])
                    u_stats, info = self.agent.update(processed_path)

                    self.memory.update_priorities(paths["idxes"], info["td_err"])
                    self.callback.add_update_info(u_stats)
                    self.callback.add_path_info(path_info, extra_info, flag='train')

                if self.callback.num_batches() >= self.print_every:
                    count = self.callback.print_table()

            for paths, path_info, extra_info in self.data_generator(1, use_noise=False):
                if paths is not None:
                    self.callback.add_path_info(path_info, extra_info, flag='val')

            if self.save_every is not None and count % self.save_every == 0:
                np.save(self.env.name + '_' + self.agent.name, self.callback.scores)
