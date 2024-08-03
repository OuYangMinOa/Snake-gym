from Snake.snake  import SNAKE as snake1
from Snake.snake2 import SNAKE as snake2
from dataclasses import dataclass

from config.config_base import ConfigBase
from model.model1 import model1, CustomActorCriticPolicy2



class configTest(ConfigBase):
    TB_LOG_NAME   = "Test"
    SNAKE_OBJ = snake2
    ENV_NAME  = "snake2"


class config1(ConfigBase):
    TB_LOG_NAME   = "base"
    SNAKE_OBJ = snake2
    ENV_NAME  = "snake2"

class config2(ConfigBase):
    TB_LOG_NAME   = "No_obs_norm"
    SNAKE_OBJ = snake2
    ENV_NAME  = "snake2"
    USE_OBS_NORM  = False

class config3(ConfigBase):
    TB_LOG_NAME   = "use_rew_norm"
    SNAKE_OBJ     = snake2
    ENV_NAME      = "snake2"
    USE_REW_NORM  = True
    USE_REW_SCALE = False


class config4(ConfigBase):
    TB_LOG_NAME   = "No_obs_norm_no_rew_any"
    SNAKE_OBJ = snake2
    ENV_NAME  = "snake2"

    USE_REW_NORM  = False
    USE_REW_SCALE = False
    USE_OBS_NORM  = False



class config5(ConfigBase):
    TB_LOG_NAME   = "No_obs_norm_rew_norm"
    SNAKE_OBJ = snake2
    ENV_NAME  = "snake2"

    USE_REW_NORM  = True
    USE_REW_SCALE = False
    USE_OBS_NORM  = False


class config6(ConfigBase):
    TB_LOG_NAME   = "use_rew_norm_scale_1"
    SNAKE_OBJ     = snake2
    ENV_NAME      = "snake2"
    USE_REW_NORM  = True
    USE_REW_SCALE = False

    REWARD_SCALE = 1

