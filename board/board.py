from stable_baselines3.common.callbacks import BaseCallback


class TCB1(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        envs = self.training_env.venv.venv.envs
        mean_body_len = 0
        mean_score    = 0
        for each in envs:
            mean_body_len += len(each.body)
            mean_score    += each.score


        self.logger.record("mean_body_len", mean_body_len/len(envs))
        self.logger.record("mean_score"   ,    mean_score/len(envs))

        return True