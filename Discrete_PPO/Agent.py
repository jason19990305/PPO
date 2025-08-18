from Discrete_PPO.ReplayBuffer import ReplayBuffer
from Discrete_PPO.ActorCritic import Actor,Critic
from Discrete_PPO.Normalization import Normalization
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

class Agent():
    def __init__(self , args , env , hidden_layer_list=[64,64]):
        # Hyperparameter
        self.max_train_steps = args.max_train_steps
        self.evaluate_freq_steps = args.evaluate_freq_steps
        self.mini_batch_size = args.mini_batch_size
        self.use_state_norm = args.use_state_norm
        self.num_actions = args.num_actions
        self.num_states = args.num_states
        self.entropy_coef = args.entropy_coef        
        self.batch_size = args.batch_size        
        self.epsilon = args.epsilon
        self.epochs = args.epochs                
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.gae = args.gae
        self.lr = args.lr     
        
        # Variable
        self.episode_count = 0
        self.total_steps = 0
        self.evaluate_count = 0

                
        # other
        self.env = env
        self.env_eval = copy.deepcopy(env)
        self.replay_buffer = ReplayBuffer(args)
        self.state_norm = Normalization(shape=self.num_states)
        self.state_norm_target = Normalization(shape=self.num_states)

        
        # The model interacts with the environment and gets updated continuously
        self.actor = Actor(args , hidden_layer_list.copy())
        self.critic = Critic(args , hidden_layer_list.copy())
        print(self.actor)
        print(self.critic)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        
        
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float)

        with torch.no_grad():
            s = torch.unsqueeze(state, dim=0)
            action_probability = self.actor(s).numpy().flatten()
            action = np.random.choice(self.num_actions, p=action_probability)
            log_prob = np.log(action_probability[action] + 1e-10)  # Add small value to avoid log(0)
            
        return action , log_prob

    def evaluate_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            s = torch.unsqueeze(state, dim=0)
            action_probability = self.actor(s)
            action =  torch.argmax(action_probability).item()
        return action
        
    def train(self):
        time_start = time.time()
        step_reward_list = []
        step_count_list = []
        
        # Training loop
        while self.total_steps < self.max_train_steps:
            # reset environment
            s, info = self.env.reset()
            if self.use_state_norm:
                self.state_norm(copy.deepcopy(s) , update = True) # update state normalization
                s = self.state_norm_target(s , update=False) # get normalized state

            while True:                
                a , log_prob = self.choose_action(s)
                # interact with environment
                s_ , r , done, truncated, _ = self.env.step(a)   
                if self.use_state_norm:
                    self.state_norm(copy.deepcopy(s_) , update = True) # update state normalization
                    s_ = self.state_norm_target(s_ , update=False) # get normalized state
                
                done = done or truncated
                # stoare transition in replay buffer
                self.replay_buffer.store(s, a , log_prob, [r], s_, [done] , [truncated | done])
                # update state
                s = s_
                
                if self.replay_buffer.count >= self.batch_size:
                    self.update()
            
                if self.total_steps % self.evaluate_freq_steps == 0:
                    self.evaluate_count += 1
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
                    
                self.total_steps += 1
                if done or truncated :
                    break

        # Plot the training curve
        plt.plot(step_count_list, step_reward_list)
        plt.xlabel("Epoch")
        plt.ylabel("Steps")
        plt.title("Training Curve")
        plt.show()
        
    def GAE(self,vs,vs_,r,done , truncated):
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            
            deltas = r + self.gamma * (1.0 - done) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(truncated.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            
        return v_target , adv
    
    def update(self):
        s, a , old_log_prob, r, s_, done , truncated = self.replay_buffer.numpy_to_tensor()
        
        
        a = a.view(-1, 1)  # Reshape action from (N) -> (N, 1) for gathering
        old_log_prob = old_log_prob.view(-1, 1)  # Reshape old_log_prob from (N) -> (N, 1) for gathering
        
        with torch.no_grad():
            # current value
            value = self.critic(s)    
            # next value        
            next_value = self.critic(s_)
            
            if self.gae:
                # Use GAE for advantage estimation
                target_value , adv = self.GAE(value, next_value, r, done , truncated)
            else :                 
                target_value = r + self.gamma * next_value * (1.0 - done)  # TD target
                adv = target_value - value
                
            # Advantage normalization
            adv = ((adv - adv.mean()) / (adv.std() + 1e-8)) 
       
        
        
        
        for i in range(self.epochs):
            for j in range(self.batch_size//self.mini_batch_size):
                index = np.random.choice(self.batch_size, self.mini_batch_size, replace=False)
                
                # Update Actor
                new_prob = self.actor(s[index]).gather(dim=1, index=a[index])  # Get action probability from the model
                log_prob = torch.log(new_prob + 1e-10)  # Add small value to avoid log(0)
                prob_entropy = -(new_prob * log_prob).mean() # get entropy of actor distribution
                ratio = torch.exp(log_prob - old_log_prob[index])  # Calculate the ratio of new and old probabilities                
                p1 = ratio * adv[index]
                p2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = torch.min(p1, p2)  - prob_entropy * self.entropy_coef # Mean loss over the batch
                actor_loss = -actor_loss.mean()
                self.optimizer_actor.zero_grad()
                actor_loss.backward()  # Backpropagation
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()
                
                # Update Critic
                value = self.critic(s[index])
                critic_loss = F.mse_loss(value, target_value[index])  # Mean Squared Error loss
                self.optimizer_critic.zero_grad()
                critic_loss.backward()  # Backpropagation
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
                
        self.lr_decay(self.total_steps)  # Learning rate decay
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
                s_, r, done, truncted, _ = env.step(a)
                if self.use_state_norm:
                    s_ = self.state_norm_target(s_ , update=False)
               
                episode_reward += r
                s = s_
                
                if truncted or done:
                    break
            evaluate_reward += episode_reward

        return evaluate_reward / times
    