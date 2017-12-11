from models.agents import *
from basic_utils.data_generator import get_generator


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.callback = Callback()
        self.agent = get_agent(self.cfg)
        self.data_generator = get_generator(self.cfg)

        if self.cfg["load_model"]:
            self.agent.load_model("./save_model/" + self.cfg["ENV_NAME"] + "_" + self.cfg["agent"])

    def train(self):
        count = 1
        while True:
            self.data_generator.set_param(self.agent.get_params())

            paths, path_info, extra_info = self.data_generator.derive_data()

            u_stats, info = self.agent.update(paths)
            self.data_generator.set_info(info)
            self.callback.add_update_info(u_stats)
            self.callback.add_path_info(path_info, extra_info)

            if self.callback.num_batches() >= self.cfg['print_every']:
                count = self.callback.print_table()

            if self.cfg['save_every'] is not None and count % self.cfg["save_every"] == 0:
                self.agent.save_model('./save_model/' + self.cfg['ENV_NAME'] + '_' + self.cfg["agent"])
                np.save('./save_score/' + self.cfg['ENV_NAME'] + '_' + self.cfg["agent"], self.callback.scores)
