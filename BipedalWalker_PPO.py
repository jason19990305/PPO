import gymnasium as gym
import numpy as np
import argparse
import time

from Continuous_PPO.Agent import Agent

class main():
    def __init__(self , args):
        
        env_name = 'BipedalWalker-v3'
        env = gym.make(env_name)
        
        args.num_states = env.observation_space.shape[0]
        args.num_actions = env.action_space.shape[0]
        print(env.action_space)
        # Pring hyperparameters 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")
        
        
        self.agent = Agent(args, env , [512,512]) # hidden layer size   
        
        self.agent.train()       
        render_env = gym.make(env_name, render_mode="human")  
        for i in range(1000):
            self.agent.evaluate(render_env)
        render_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    parser.add_argument("--evaluate_freq_steps", type=float, default=2e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Mini-batch size for training")
    parser.add_argument("--max_train_steps", type=int, default=1e6, help="Set the number of steps used for training the agent")    
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--entropy_coef", type=float, default=0.00, help="Entropy coefficient for actor loss")
    parser.add_argument("--continuous", type=bool, default=False, help="Whether the action space is continuous or discrete")
    parser.add_argument("--epochs", type=int, default=10, help="PPO training iteration parameter")
    parser.add_argument("--epsilon", type=float, default=0.18, help="PPO clip parameter")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--gae", type=bool, default=False, help="Use GAE for advantage estimation")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate for optimizer")
    

    args = parser.parse_args()
    
    main(args)
