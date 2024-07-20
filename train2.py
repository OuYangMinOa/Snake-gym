from stable_baselines3 import PPO
from Snake2 import SNAKE


import time
import cv2
import os



models_dir = f"models2/{int(time.time())}/"
logdir = f"logs2/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)


env = SNAKE()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)


TIMESTEPS = 100
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")






