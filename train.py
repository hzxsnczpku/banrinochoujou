from models.agents import *
from basic_utils.data_generator import get_generator, Path_Data_generator


class Trainer:
    def __init__(self, agent, env, n_worker=10, path_num=10, save_every=None, render=False, print_every=10, load_model=False):
        self.callback = Callback()
        self.save_every = save_every
        self.print_every = print_every

        self.agent = agent
        self.env = env
        self.data_generator = Path_Data_generator(self.agent, self.env, n_worker, path_num, render)

        if load_model:
            self.agent.load_model("./save_model/" + self.env.name + "_" + self.agent.name)

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
