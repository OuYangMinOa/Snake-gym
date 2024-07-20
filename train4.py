from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from model2 import CustomActorCriticPolicy
from Snake3 import SNAKE

import time
import os

models_dir = f"models3/{int(time.time())}/"
logdir = f"logs3/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)


env = SNAKE()
# env = SubprocVecEnv( [SNAKE for _ in range(5)],start_method="forkserver")

model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log=logdir)


TIMESTEPS = 1000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
