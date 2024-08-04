from Snake.snake  import SNAKE as snake1
from Snake.snake2 import SNAKE as snake2
from Snake.snake3 import SNAKE as snake3

from dataclasses import dataclass

from config.config_base import ConfigBase
from model.model1 import model1, model1_2, model1_3



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


class config7(ConfigBase):
    TB_LOG_NAME   = "No_obs_norm_rew_norm"
    MODEL = model1_2
    MODEL_NAME = "MODEL1_2"
    SNAKE_OBJ = snake2
    ENV_NAME  = "snake2"

    USE_REW_NORM  = True
    USE_REW_SCALE = False
    USE_OBS_NORM  = False

class config8(ConfigBase):
    TB_LOG_NAME   = "REWARD_SCALE_p2"
    MODEL = model1_3
    MODEL_NAME = "MODEL1_3"
    SNAKE_OBJ = snake2
    ENV_NAME  = "snake2"

    REWARD_SCALE = 0.2


class config9(ConfigBase):
    TB_LOG_NAME   = "no_rew_norm"
    MODEL = model1_2
    MODEL_NAME = "MODEL1_2"
    SNAKE_OBJ = snake2
    ENV_NAME  = "snake2"

    USE_REW_NORM  = False
    USE_REW_SCALE = False
    USE_OBS_NORM  = True
    ENT_COEF      = 0


class config10(ConfigBase):
    TB_LOG_NAME   = "no_norm"
    MODEL = model1_2
    MODEL_NAME = "MODEL1_2"
    SNAKE_OBJ = snake2
    ENV_NAME  = "snake2"

    USE_REW_NORM  = False
    USE_REW_SCALE = False
    USE_OBS_NORM  = False
    ENT_COEF      = 0.0


class config11(ConfigBase):
    TB_LOG_NAME   = "No_obs_norm_rew_norm"
    MODEL = model1_3
    MODEL_NAME = "MODEL1_3"
    SNAKE_OBJ = snake2
    ENV_NAME  = "snake2"

    USE_REW_NORM  = True
    USE_REW_SCALE = False
    USE_OBS_NORM  = False


class config12(ConfigBase):
    TB_LOG_NAME   = "No_obs_norm_rew_norm"
    MODEL = model1_2
    MODEL_NAME = "MODEL1_2"
    SNAKE_OBJ = snake3
    ENV_NAME  = "snake3"

    USE_REW_NORM  = True
    USE_REW_SCALE = False
    USE_OBS_NORM  = False