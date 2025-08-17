from gymnasium.vector import AsyncVectorEnv
import gymnasium as gym
import numpy as np
import os

def main():
    num_envs = 8
    env_name = 'Acrobot-v1'
    env_fns = [lambda : gym.make(env_name) for _ in range(num_envs)]
    
    
    venv = AsyncVectorEnv(env_fns , autoreset_mode= gym.vector.AutoresetMode.SAME_STEP)
    
    total_steps = 501

    print("Observation Space : ",venv.single_observation_space.shape[0])
    print("Action Space : ",venv.single_action_space.n)
    
    s_list = []
    a_list = []
    r_list = []
    s__list = []
    done_list = []
    truncated_list = []
    
    s, infos = venv.reset()              # obs shape: (num_envs, obs_dim)

    for step in range(total_steps):
        # a = choose_action(s)  # Replace with your action selection logic
        actions = np.array([venv.single_action_space.sample() for _ in range(num_envs)],dtype=venv.single_action_space.dtype)
        
        s_, r, done, truncated, infos = venv.step(actions)
        for i in range(num_envs):
            if done[i] or truncated[i]:
                next_state = infos["final_obs"][i]
            else : 
                next_state = s_[i]
            s_list.append(s[i])
            a_list.append(actions[i])
            r_list.append(r[i])
            s__list.append(next_state)
            done_list.append(done[i])
            truncated_list.append(truncated[i])
            if i == 0:
                print("------------")
                print("State:", s[i])
                print("Next State:", next_state)
                print("Truncated:", truncated[i])
        s = s_
    a_list = np.array(a_list)
    s_list = np.array(s_list)
    r_list = np.array(r_list)
    s__list = np.array(s__list)
    done_list = np.array(done_list)
    truncated_list = np.array(truncated_list)
    print("s_list shape:", s_list.shape)
    print("a_list shape:", a_list.shape)
    print("r_list shape:", r_list.shape)
    print("s__list shape:", s__list.shape)
    print("done_list shape:", done_list.shape)
    print("truncated_list shape:", truncated_list.shape)
    print(venv.autoreset_mode)
        
if __name__ == "__main__":
    main()