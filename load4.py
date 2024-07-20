
# laod 

from stable_baselines3 import PPO
from Snake3 import SNAKE
from model2 import CustomActorCriticPolicy


import os
import time
import cv2
import glob

env = SNAKE()



model_name = r"\1721466221"

while True:
     total  = glob.glob( r"D:\python\snake_gym\models3" + model_name + r"\*.zip")
     
     model_path = r"D:\python\snake_gym\models3" + f"\\{model_name}\\{len(total)*1000}.zip" 
     print(model_path)

     model = PPO.load(model_path,env=env)    

     obs, info = env.reset()
     done = False
     while not done:
          
          
          action, _ = model.predict(obs)
          obs, reward, done, _, info = env.step(action)
          
          env.render()
          if cv2.waitKey(30) & 0xFF == ord('q'):
               break
          
          # time.sleep(0.1)

env.close()
