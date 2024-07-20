# laod 

from stable_baselines3 import PPO
import os
from Snake2 import SNAKE
import time
import cv2
import glob

env = SNAKE()




while True:
     total  =glob.glob( r"D:\python\snake_gym\models2\1659127717\*.zip")
     
     model_path = r"D:\python\snake_gym\models2\1659127717" + f"\\{len(total)*100}.zip" 
     print(model_path)

     model = PPO.load(model_path,env=env)    

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
