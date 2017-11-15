import time
from basic_utils.layers import mujoco_layer_designer
from models.agents import *
from basic_utils.data_generator import Whole_path_Data_generator


class Asy_train:
    def __init__(self, cfg):
        self.cfg = update_default_config(MLP_OPTIONS, cfg)
        self.cfg = get_env_info(self.cfg)
        if self.cfg["use_mujoco_setting"]:
            self.cfg = mujoco_layer_designer(self.cfg)

        self.callback = Callback()
        self.agent, self.cfg = get_agent(self.cfg)
        self.data_generator = Whole_path_Data_generator(self.cfg)

        if self.cfg["load_model"]:
            self.agent.load_model("./save_model/" + self.cfg["ENV_NAME"] + "_" + self.cfg["agent"])

    def train(self):
        tstart = time.time()
        while True:
            self.data_generator.set_param(self.agent.get_params())
            paths, stats, scores = self.data_generator.derive_data()

            for u in self.agent.update(paths):
                add_prefixed_stats(stats, u[0], u[1])
            stats["TimeElapsed"] = time.time() - tstart
            counter = self.callback(stats)

            if self.cfg['save_every'] is not None and counter % self.cfg["save_every"] == 0:
                self.agent.save_model('./save_model/' + self.cfg['ENV_NAME'] + '_' + self.cfg["agent"])
                np.save('./save_score/' + self.cfg['ENV_NAME'] + '_' + self.cfg["agent"], scores)
