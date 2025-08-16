import gymnasium as gym
import numpy as np
import argparse
import time

from Discrete_PPO.Agent import Agent

class main():
    def __init__(self , args):
        
        env_name = 'Acrobot-v1'
        env = gym.make(env_name)
        
        args.num_states = env.observation_space.shape[0]
        args.num_actions = env.action_space.n
        
        # Pring hyperparameters 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")
        
        
        
        self.agent = Agent(args, env , [128,128]) # hidden layer size   
        
        self.agent.train()       
        render_env = gym.make(env_name, render_mode="human")  
        for i in range(1000):
            self.agent.evaluate(render_env)
        render_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    parser.add_argument("--evaluate_freq_steps", type=float, default=2e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="Mini-batch size for training")
    parser.add_argument("--max_train_steps", type=int, default=1e5, help="Set the number of steps used for training the agent")    
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Whether the action space is continuous or discrete")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient for actor loss")
    parser.add_argument("--continuous", type=bool, default=False, help="Whether the action space is continuous or discrete")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--epochs", type=int, default=50, help="PPO training iteration parameter")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    

    args = parser.parse_args()
    
    main(args)
