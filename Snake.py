import gym
from gym import spaces
import numpy as np
import cv2
from PIL import Image
import time
from collections import deque


ENV_X = 20
ENV_Y = 20
MAX_SNAKE_LENGTH = 30

class SNAKE(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SNAKE, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=ENV_Y,shape=(9,), dtype=np.float64)

    def snake_len(self):
        return len(self.body)

    def step(self, action):


        if (action==0):
            self.direction += 3
        if (action==2):
            self.direction += 1

        self.direction %= 4
        next_x, next_y = self.body[0]

        if (self.direction %2==0):
            next_x += (self.direction-1)
        else:
            next_y += (2-self.direction)
        self.body = [(next_x,next_y),] + self.body[0:self.snake_len()-1]

        head_x, head_y = self.body[0]

        reward  = 0
        reward -= self.steps
        reward -= np.sqrt( (head_x-self.apple_x)**2+(head_y-self.apple_y)**2) / 10

        if (head_x <0 or head_x >=ENV_X):
            self.done = True
            reward = -100
        if (head_y <0 or head_y >=ENV_Y):
            self.done = True
            reward = -100

        for i in self.body[1:]:
            if (head_x,head_y) == i:
                self.done = True
                reward = -100


        if (head_x==self.apple_x and head_y==self.apple_y):
            self.score   += 1
            reward  += 50 *self.score
            tail = self.body[-1]
            self.body.append(tail)
            self.apple()

        # if (self.steps==1000):
        #     done = False

        info = {}
        return self.get_state(), reward, self.done, info

    def get_state(self):
        head_x, head_y = self.body[0]
        up_y     , down_y = head_y, ENV_Y - head_y
        left_x  , right_x = head_x, ENV_X - head_x


        for bx,by in self.body[1:]:
            if (bx == head_x):
                distance = by - head_y
                if (distance > 0 and distance < up_y):
                    up_y = distance
                elif (distance<down_y):
                    down_y = distance
            elif (by == head_y):
                distance = bx - head_x
                if (distance > 0 and distance < right_x):
                    right_x = distance
                elif (distance < left_x):

                    left_x = distance
        return np.array([self.direction, head_x, head_y, self.apple_x, self.apple_y, up_y, down_y, left_x  , right_x])


    def apple(self):
        self.apple_x = np.random.randint(0,ENV_X-1)
        self.apple_y = np.random.randint(0,ENV_Y-1)
        while (self.apple_x ,self.apple_y) in self.body:
            self.apple_x = np.random.randint(0,ENV_X-1)
            self.apple_y = np.random.randint(0,ENV_Y-1)

    def fill_pass(self):
        for i in range(self.snake_len()):
            if (i== MAX_SNAKE_LENGTH):
                break
            self.path_body[i] == self.body[i]

    def reset(self):
        self.x    = ENV_X//2
        self.y    = ENV_Y//2
        self.body = [(self.x,self.y),(self.x,self.y-1),(self.x,self.y-2)]
        self.direction = 0
        self.action    = 0
        self.done      = False
        self.score     = 0
        self.steps     = 0
        self.path_body = deque(self.body, maxlen=MAX_SNAKE_LENGTH)

        for i in range(MAX_SNAKE_LENGTH-3):
            self.path_body.append(-1)


        #  randomize apple
        self.apple()
        # self.fill_pass()

        return self.get_state()  # reward, done, info can't be included


    def render(self, mode='human'):
        SNAKE_COLOR = (255,255,255)
        SNAKE_COLOR_HEAD  = (0,255,0)
        APPLE_COLOR = (0,0,255)

        env = np.zeros((ENV_X*30, ENV_Y*30,3), dtype=np.uint8)
        # env[self.apple_x*30:(self.apple_x+1)*30,self.apple_y*30:(self.apple_y+1)*30] = APPLE_COLOR


        cv2.circle(env,
              (self.apple_x*30+15,self.apple_y*30+15),15
            ,(0,0,255),-1)



        cv2.rectangle(env,(self.body[0][0]*30,self.body[0][1]*30),(self.body[0][0]*30+30,self.body[0][1]*30+30),SNAKE_COLOR_HEAD,8)
        for position in self.body[1:]:
            cv2.rectangle(env,(position[0]*30,position[1]*30),(position[0]*30+30,position[1]*30+30),SNAKE_COLOR,8)


        cv2.imshow('img',env)
        



        t_end = time.time() + 0.05
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue


            


    def close (self):
        self.done = False




if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = SNAKE()

    check_env(env)