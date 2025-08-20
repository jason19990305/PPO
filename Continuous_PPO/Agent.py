from Continuous_PPO.ReplayBuffer import ReplayBuffer
from Continuous_PPO.ActorCritic import Actor,Critic
from Continuous_PPO.Normalization import Normalization
from gymnasium.vector import AsyncVectorEnv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import copy
import time
import os 

class Agent():
    def __init__(self , args , env , hidden_layer_list=[64,64]):
        # Hyperparameter
        self.max_train_steps = args.max_train_steps
        self.evaluate_freq_steps = args.evaluate_freq_steps
        self.mini_batch_size = args.mini_batch_size
        self.use_state_norm = args.use_state_norm
        self.num_actions = args.num_actions
        self.num_states = args.num_states
        self.env_name = args.env_name
        self.entropy_coef = args.entropy_coef
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.epochs = args.epochs        
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.gae = args.gae
        self.lr = args.lr     
        
        # Variable
        self.total_steps = 0

                
        # other
        self.env = env
        self.env_eval = copy.deepcopy(env)
        self.num_envs = os.cpu_count() - 1
        self.replay_buffer = ReplayBuffer(args)
        self.state_norm = Normalization(shape=self.num_states)
        self.state_norm.running_ms.name = "state_norm"
        self.state_norm_target = Normalization(shape=self.num_states)
        self.state_norm_target.running_ms.name = "state_norm_target"
        
        env_fns = [lambda : gym.make(self.env_name) for _ in range(self.num_envs)]
        self.venv = AsyncVectorEnv(env_fns , autoreset_mode= gym.vector.AutoresetMode.SAME_STEP)
        
        
        # The model interacts with the environment and gets updated continuously
        self.actor = Actor(args , hidden_layer_list.copy())
        self.critic = Critic(args , hidden_layer_list.copy())
        print(self.actor)
        print(self.critic)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        
    def __del__(self):
        if hasattr(self, 'venv') and self.venv is not None:
            self.venv.close()
            print("VectorEnv closed in __del__.")

    # choose action
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float)

        with torch.no_grad():
            dist = self.actor.get_dist(state)
            a = dist.sample()
            a = torch.clamp(a, -1, 1)  # Ensure action is within bounds
            log_prob = dist.log_prob(a)
            
        return a.numpy() , log_prob.numpy()
    
    # evaluate action
    def evaluate_action(self, state):
        state = torch.tensor(state, dtype=torch.float)

        with torch.no_grad():
            s = torch.unsqueeze(state, dim=0)
            a = self.actor(s)
            a = torch.clamp(a, -1, 1)  # Ensure action is within bounds

        return a.numpy().flatten()
    
    def train(self):
        time_start = time.time()
        step_reward_list = []
        step_count_list = []
        evaluate_count = 0
        
        # Reset Vector Env
        s , infos = self.venv.reset()        
        # State Normalization
        if self.use_state_norm:        
            for i in range(self.num_envs):           
                self.state_norm(s[i] , update = True)
                s[i] = self.state_norm_target(s[i] , update=False) # get normalized state
                

        # Training Loop
        for i in range(int(self.max_train_steps//self.batch_size)):
            # Sample data
            for step in range(self.batch_size // self.num_envs + 1):
                # Choose action
                a , log_prob = self.choose_action(s)
                # Scale action
                action = a * self.env.action_space.high[0]
                s_ , r , done, truncated, infos = self.venv.step(action)  # Vector Env
                # Handle final state
                for j in range(self.num_envs):
                    if done[j] or truncated[j]:
                        next_state = infos["final_obs"][j]
                    else : 
                        next_state = copy.deepcopy(s_[j])
                    # State Normalization
                    if self.use_state_norm:
                        self.state_norm(next_state , update = True) # update state normalization                        
                        next_state = self.state_norm_target(next_state , update=False) # get normalized state
                        s_[j] = self.state_norm_target(s_[j] , update=False) # get normalized state
                        

                    # s, a , log_prob , r, s_, done , truncate
                    self.replay_buffer.store(s[j], a[j], log_prob[j], [r[j]], next_state, [done[j]], [truncated[j] or done[j]])
                    self.total_steps += 1
                    evaluate_count += 1
                
                s = s_
            
            # Update 
            self.update()
            
            # Evaluate
            if evaluate_count >= self.evaluate_freq_steps:
                evaluate_reward = self.evaluate(self.env_eval)
                step_reward_list.append(evaluate_reward)
                step_count_list.append(self.total_steps)
                time_end = time.time()
                h = int((time_end - time_start) // 3600)
                m = int(((time_end - time_start) % 3600) // 60)
                second = int((time_end - time_start) % 60)
                print("---------")
                print("Time : %02d:%02d:%02d"%(h,m,second))
                print("Step : %d / %d\tEvaluate reward : %0.2f"%(self.total_steps,self.max_train_steps,evaluate_reward))
                evaluate_count = 0
                
        # Plot the training curve
        plt.plot(step_count_list, step_reward_list)
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Training Curve")
        plt.show()
                
    
    
    
    def update(self):
        s, a, old_log_prob , r, s_, done , truncated = self.replay_buffer.numpy_to_tensor()
        
        with torch.no_grad():
            # current value
            value = self.critic(s)    
            # next value        
            next_value = self.critic(s_)
            
            if self.gae:
                # Use GAE for advantage estimation
                target_value , adv = self.GAE(value, next_value, r, done , truncated)
            else :                 
                # TD-Error
                target_value = r + self.gamma * next_value * (1.0 - done)  
                # baseline
                adv = target_value - value
            
            # advantage normalization
            adv = ((adv - adv.mean()) / (adv.std() + 1e-8))
       
        
        batch_size = s.shape[0]
        for i in range(self.epochs):
            for j in range(batch_size//self.mini_batch_size):
                # random sample
                index = np.random.choice(batch_size, self.mini_batch_size, replace=False)
                
                # Update Actor
                # Get action probability from the model
                dist = self.actor.get_dist(s[index])
                # get entropy of actor distribution
                prob_entropy = dist.entropy().sum(dim=1, keepdim=True)                   
                # Get log probability
                log_prob = dist.log_prob(a[index])
                # Calculate the ratio of new and old probabilities
                ratio = torch.exp(log_prob.sum(dim=1, keepdim=True) - old_log_prob[index].sum(dim=1, keepdim=True))
                p1 = ratio * adv[index]
                p2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = torch.min(p1, p2) - prob_entropy * self.entropy_coef 
                
                # Mean actor loss and add entropy term
                actor_loss = -actor_loss.mean()
                self.optimizer_actor.zero_grad()
                actor_loss.backward()  # Backpropagation
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()
                
                # Update Critic
                value = self.critic(s[index])
                critic_loss = F.mse_loss(value, target_value[index])  # Mean Squared Error loss
                self.optimizer_critic.zero_grad()
                critic_loss.backward()  # Backpropagation
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
        # Update learning rate
        self.lr_decay(self.total_steps)  # Learning rate decay
        # Update state normalization
        self.update_normalization()
        
    def update_normalization(self):
        if self.use_state_norm:
            self.state_norm_target.running_ms.mean = self.state_norm.running_ms.mean
            self.state_norm_target.running_ms.std = self.state_norm.running_ms.std
            
    def lr_decay(self, total_steps):
        lr_a_now = self.lr * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr * (1 - total_steps / self.max_train_steps)
        for opt in self.optimizer_actor.param_groups:
            opt['lr'] = lr_a_now
        for opt in self.optimizer_critic.param_groups:
            opt['lr'] = lr_c_now
        
    def evaluate(self , env):
        times = 10
        evaluate_reward = 0
        
        for i in range(times):
            s , info = env.reset()
            if self.use_state_norm:
                s = self.state_norm_target(s , update=False)
                
            episode_reward = 0
            while True:
                a = self.evaluate_action(s)  # We use the deterministic policy during the evaluating
                action = a * self.env.action_space.high[0]
                s_, r, done, truncted, _ = env.step(action)
                if self.use_state_norm:
                    s_ = self.state_norm_target(s_ , update=False)

               
                episode_reward += r
                s = s_
                
                if truncted or done:
                    break
            evaluate_reward += episode_reward

        return evaluate_reward / times
    