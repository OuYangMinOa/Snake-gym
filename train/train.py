## train config

from stable_baselines3.common.vec_env  import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.vec_env  import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env  import DummyVecEnv
from stable_baselines3                 import PPO

import time

from config.config_base import ConfigBase
from glob import glob


class Trainler:
    def __init__(self, config : ConfigBase) -> None:
        self.config = config


    def make_env(self):
        def _init():
            env                    = self.config.SNAKE_OBJ()
            env.use_reward_scaling = self.config.REWARD_SCALE
            env.apple_reward       = self.config.APPLE_REWARD
            env.reset()
            return env
        return _init

    def get_last_model(self,):
        total = glob( f"{self.config.MODEL_DIR}"  + r"\*.zip")
        if (len(total)==0):
            return None
        model_path = f"{self.config.MODEL_DIR}"  + f"\\{len(total)*self.config.TIME_STEPS}.zip" 
        return model_path

    def build_env(self):
        env = DummyVecEnv([self.make_env() for _ in range(self.config.ENV_NUMS)])
        env = VecMonitor(env)
        env = VecNormalize( venv        = env, 
                            norm_obs    = self.config.USE_OBS_NORM,
                            norm_reward = self.config.USE_REW_NORM,
                            clip_obs    = self.config.CLIP_OBS,
                            gamma       = self.config.GAMMA,)
        return env
    
    def build_single_env(self):
        env = DummyVecEnv([self.make_env(),])
        env = VecNormalize( venv         = env, 
                            norm_obs    = self.config.USE_OBS_NORM,
                            norm_reward = self.config.USE_REW_NORM,
                            clip_obs    = self.config.CLIP_OBS,
                            gamma       = self.config.GAMMA,)
        return env

    def build_model(self):
        model_path = self.get_last_model()
        if (model_path):
            print(f"Load model : {model_path}")
            model = PPO.load(path           = model_path, 
                            env             = self.build_env(), 
                            gamma           = self.config.GAMMA,
                            n_steps         = self.config.N_STEPS,
                            tensorboard_log = self.config.LOG_DIR,
                            ent_coef        = self.config.ENT_COEF,
                            seed            = self.config.MODEL_SEED,
                            learning_rate   = self.config.LEARN_RATE,
                            batch_size      = self.config.BATCH_SIZE, 
                            verbose         = 1,)
        else:
            print(f"Create new model")
            model = PPO(policy          = self.config.MODEL, 
                        env             = self.build_env(), 
                        gamma           = self.config.GAMMA,
                        n_steps         = self.config.N_STEPS,
                        tensorboard_log = self.config.LOG_DIR,
                        ent_coef        = self.config.ENT_COEF,
                        seed            = self.config.MODEL_SEED,
                        learning_rate   = self.config.LEARN_RATE,
                        batch_size      = self.config.BATCH_SIZE, 
                        verbose         = 1,)
        return model
    
    def learn(self):
        iters = 0
        model = self.build_model()
        while True:
            iters += 1
            model.learn(total_timesteps     = self.config.TIME_STEPS,
                        tb_log_name         = self.config.TB_LOG_NAME,
                        reset_num_timesteps = False,  
                        callback            = self.config.TBC_OBJ(),
                        )
            model.save(f"{self.config.MODEL_DIR}/{self.config.TIME_STEPS*iters}")

    def test(self):
        import cv2

        while True:
            env   = self.build_single_env()
            env   = env.unwrapped.envs[0]
            model = self.build_model()
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True )
                obs, reward, _, done, _ = env.step(action)
                print(action, reward, done)
                env.render()

                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

                time.sleep(1/60)

                
        



