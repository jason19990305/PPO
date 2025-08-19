import gymnasium as gym
import numpy as np
import argparse

from Continuous_PPO.Agent import Agent

from gymnasium.wrappers import RecordVideo

class main():
    def __init__(self , args):
        
        env_name = 'Walker2d-v5'
        # Create the environment for evaluation
        env = gym.make(env_name)
        
        args.num_states = env.observation_space.shape[0]
        args.num_actions = env.action_space.shape[0]
        args.env_name = env_name
        print(env.action_space)
        # Pring hyperparameters 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")
        
        
        self.agent = Agent(args, env , [512,512,512]) # hidden layer size   
        
        
        self.agent.train()       
        render_env = gym.make(env_name, render_mode="rgb_array")  
        render_env = RecordVideo(render_env, video_folder = "Video/"+env_name, episode_trigger=lambda x: True)
        self.agent.evaluate(render_env)
        render_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    parser.add_argument("--evaluate_freq_steps", type=float, default=2e3, help="Evaluate the policy every N environment steps")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="Mini-batch size used for PPO updates")
    parser.add_argument("--max_train_steps", type=int, default=3e6, help="Total number of environment steps for training")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Whether to apply state normalization")
    parser.add_argument("--entropy_coef", type=float, default=0.00, help="Entropy bonus coefficient added to the actor loss")
    parser.add_argument("--batch_size", type=int, default=4096, help="Number of collected samples per update (batch size)")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clipping parameter (epsilon)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to run over the batch each update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards (gamma)")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE lambda parameter (bias–variance trade-off)")
    parser.add_argument("--gae", type=bool, default=False, help="Use GAE for advantage estimation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizers")
    

    args = parser.parse_args()
    
    main(args)
