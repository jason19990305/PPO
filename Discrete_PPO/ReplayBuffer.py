import numpy as np 
import torch

class ReplayBuffer:
    def __init__(self, args):
        self.clear_batch()
        
    def clear_batch(self):
        self.s = []
        self.a = []
        self.log_prob = []
        self.r = []
        self.s_ = []
        self.done = []
        self.truncated = []
        self.count = 0
        
    def store(self, s, a , log_prob , r, s_, done , truncated):
        self.s.append(s)
        self.a.append(a)
        self.log_prob.append(log_prob)
        self.r.append(r)
        self.s_.append(s_)
        self.done.append(done)
        self.truncated.append(truncated)
        self.count += 1
        
    def numpy_to_tensor(self):
        s = torch.tensor(np.array(self.s), dtype=torch.float)
        a = torch.tensor(np.array(self.a), dtype=torch.int64)
        log_prob = torch.tensor(np.array(self.log_prob), dtype=torch.float)
        r = torch.tensor(np.array(self.r), dtype=torch.float)
        s_ = torch.tensor(np.array(self.s_), dtype=torch.float)
        done = torch.tensor(np.array(self.done), dtype=torch.float)
        truncated = torch.tensor(np.array(self.truncated), dtype=torch.float)
        self.clear_batch()

        return s, a , log_prob , r, s_, done , truncated
