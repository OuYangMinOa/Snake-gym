from time import sleep
import argparse


from train.train import Trainler


from config.config_base import ConfigBase
from config.configs import *

# config
parser = argparse.ArgumentParser()
parser.add_argument('number', type=int, help='The `number` of config ( config{number} )')
args = parser.parse_args()


this_config = config1()
config_str  = f"this_config = config{args.number}()"
exec(config_str)
print(this_config)


tlr  = Trainler(this_config)

def train():
    print("Start train in 3 seconds")
    sleep(3)
    tlr.learn()

def test():
    print("Start train")
    tlr.test()


if __name__=="__main__":
    train()

    # print( exec("config1()"))