from torch.distributions import Normal
import torch.nn as nn
import torch

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    
class Actor(nn.Module):
    def __init__(self, args, hidden_layers=[64, 64]):
        super(Actor, self).__init__()

        self.num_states = args.num_states
        self.num_actions = args.num_actions

        # Insert input and output sizes into hidden_layers
        hidden_layers.insert(0, self.num_states)
        hidden_layers.append(self.num_actions)

        # Create fully connected layers
        fc_list = []
        for i in range(len(hidden_layers) - 1):
            num_input = hidden_layers[i]
            num_output = hidden_layers[i + 1]
            layer = nn.Linear(num_input, num_output)
            fc_list.append(layer)
            orthogonal_init(fc_list[-1])
        orthogonal_init(fc_list[-1], gain=0.01)

        # Convert list to ModuleList for proper registration
        self.layers = nn.ModuleList(fc_list)
        self.log_std = nn.Parameter(torch.zeros(1, self.num_actions))
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Pass input through all layers except the last, applying ReLU activation
        for i in range(len(self.layers)):
            x = self.tanh(self.layers[i](x))
        return x
    
    def get_dist(self,state):
        mean = self.forward(state)
        std = torch.exp(self.log_std.expand_as(mean))
        try:
            dist = Normal(mean,std)
        except Exception as e:
            for param in self.parameters():
                print("actor parameter:",param.data)
            
        return dist
    
class Critic(nn.Module):
    def __init__(self, args,hidden_layers=[64,64]):
        super(Critic, self).__init__()
        self.num_states = args.num_states
        self.num_actions = args.num_actions
        # add in list
        hidden_layers.insert(0,self.num_states)
        hidden_layers.append(1)
        print(hidden_layers)

        # create layers
        fc_list = []

        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)            
            fc_list.append(layer)
            orthogonal_init(fc_list[-1])
        # put in ModuleList
        self.layers = nn.ModuleList(fc_list)
        self.tanh = nn.Tanh()

    def forward(self,x):
        
        for i in range(len(self.layers)-1):
            x = self.tanh(self.layers[i](x))

        # predicet value
        v_s = self.layers[-1](x)
        return v_s
    
    