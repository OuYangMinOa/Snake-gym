# laod 

from stable_baselines3 import PPO
import os
from Snake import SNAKE
import time
import cv2

env = SNAKE()

model_path = r"D:\python\snake_gym\models\1659099571\3303900.zip"
model = PPO.load(model_path,env=env)



for ep in range(10):
     obs = env.reset()
     done = False
     while not done:
          
          
          action, _ = model.predict(obs)
          obs, reward, done, info = env.step(action)
          
          env.render()
          if cv2.waitKey(30) & 0xFF == ord('q'):
               break
          
          # time.sleep(0.1)

env.close()