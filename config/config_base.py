from dataclasses import dataclass
import os

from board.board  import TCB1
from Snake.snake  import SNAKE
from model.model1 import model1

@dataclass
class ConfigBase:
    TBC_OBJ       = TCB1
    SNAKE_OBJ     = SNAKE
    MODEL         = model1
    DIR_HEAD      = "train_snake_data/"
    MODEL_NAME    = "MODEL1"
    ENV_NAME      = "SNAKE1"
    TB_LOG_NAME   = "base"
    GAMMA         = 0.99
    USE_OBS_NORM  = True
    USE_REW_NORM  = False
    USE_REW_SCALE = True
    N_STEPS       = 2048
    ENT_COEF      = 0.01
    TIME_STEPS    = 2000
    CLIP_OBS      = 10
    ENV_NUMS      = 5
    LEARN_RATE    = 3e-4
    BATCH_SIZE    = 64
    MODEL_SEED    = 1
    ENV_GEN_SEED  = 1

    REWARD_SCALE  = 0.1
    APPLE_REWARD  = 50
    
    @property
    def MODEL_DIR(self):
        temp_model_dir = f"{self.DIR_HEAD}/models/{self.MODEL_NAME}/{self.ENV_NAME}/{self.TB_LOG_NAME}/"
        os.makedirs(temp_model_dir, exist_ok=True)
        return temp_model_dir

    @property
    def LOG_DIR(self):
        temp_log_dir = f"{self.DIR_HEAD}/logs/{self.MODEL_NAME}/{self.ENV_NAME}/"
        os.makedirs(temp_log_dir, exist_ok=True)
        return temp_log_dir