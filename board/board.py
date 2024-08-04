from stable_baselines3.common.callbacks import BaseCallback

import numpy as np

class TCB1(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        mean_body_len = self.training_env.venv.venv.env_method("snake_len")
        mean_score    = self.training_env.venv.venv.get_attr("score")


        self.logger.record("mean_body_len", np.mean(mean_body_len))
        self.logger.record("mean_score"   , np.mean(   mean_score))

        return True