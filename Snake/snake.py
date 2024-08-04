import time
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from collections import deque
from utils.reward_sacling import RewardScaling


ENV_X = 20
ENV_Y = 20
MAX_SNAKE_LENGTH = ENV_X * ENV_Y

class SNAKE(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SNAKE, self).__init__()

        self.reward_scale = 1
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=ENV_Y,shape=(MAX_SNAKE_LENGTH * 2 + 12,), dtype=np.float64)

        self.use_reward_scaling = True
        self.reward_scaler = RewardScaling(shape = 1,gamma=0.99)
        
    def do_reward_scale(self,x):
        if (self.use_reward_scaling):
            return self.reward_scaler(x)
        return x
    
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
        # reward -= self.steps * 0.1
        reward -= np.sqrt( (head_x-self.apple_x)**2+(head_y-self.apple_y)**2) * self.reward_scale
        self.steps += 1

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
            self.steps    = 0
            reward        = 100
            tail = self.body[-1]
            self.body.append(tail)
            self.apple()

        # if (self.steps==1000):
        #     done = False
        info = {}
        return self.get_state(), self.do_reward_scale(reward), self.done, False, info

    def get_state(self):
        head_x, head_y = self.body[0]
        up_y     , down_y = head_y, ENV_Y - head_y
        left_x  , right_x = head_x, ENV_X - head_x

        for bx,by in self.body[1:]:
            if (bx == head_x):
                dy = by - head_y
                distance = abs(dy)
                if (dy > 0 and distance < up_y):
                    up_y = abs(distance)
                elif (distance<down_y):
                    down_y =  abs(distance)
            elif (by == head_y):
                dx = bx - head_x
                distance = abs(dx)
                if (dx > 0 and distance < right_x):
                    right_x = abs(distance)
                elif (distance < left_x):
                    left_x = abs(distance)

        last_30 = list(np.array(self.body).flatten())
        this_len = len(last_30)
        if ( this_len < MAX_SNAKE_LENGTH):
            last_30.extend([0 for i in range(2 * MAX_SNAKE_LENGTH-this_len)])

        return np.array(last_30 + [self.direction, len(self.body)
,head_x, head_y, self.apple_x, self.apple_y, self.body[-1][0], self.body[-1][1], up_y, down_y, left_x  , right_x] ).astype(np.float64)


    def apple(self):
        self.apple_x = np.random.randint(0,ENV_X-1)
        self.apple_y = np.random.randint(0,ENV_Y-1)
        while (self.apple_x ,self.apple_y) in self.body:
            self.apple_x = np.random.randint(0,ENV_X-1)
            self.apple_y = np.random.randint(0,ENV_Y-1)

    def reset(self,seed=None, **kwargs):
        super().reset(**kwargs)

        self.x    = ENV_X//2
        self.y    = ENV_Y//2
        self.body = [(self.x,self.y),(self.x,self.y-1),(self.x,self.y-2)]
        self.direction = 0
        self.action    = 0
        self.done      = False
        self.score     = 0
        self.steps     = 0
        self.path_body = deque( maxlen=MAX_SNAKE_LENGTH)

        # for i in range(MAX_SNAKE_LENGTH-3):
        #     self.path_body.append(-1)


        #  randomize apple
        self.apple()

        return self.get_state(), {}  # reward, done, info can't be included


    def render(self, mode='human'):
        import cv2

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

    print(env.get_state())
