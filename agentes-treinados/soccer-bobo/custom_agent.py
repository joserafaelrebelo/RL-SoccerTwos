import numpy as np
from soccer_twos import AgentInterface
from torch import nn
import torch
from torch.nn.modules.activation import Tanh

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 16)
        self.mlp = nn.Sequential(
            nn.Linear(42*17, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, s):
        e = self.emb(s[0])
        e_flat = torch.flatten(e, start_dim=1)
        x = torch.cat((e_flat, s[1]), 1)
        return self.mlp(x)

class CustomTanh(nn.Module):
    def forward(self, input):
        return 1.6*torch.tanh(input)

class Q_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp0 = nn.Sequential(
            nn.Linear(2*128, 128, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 9, bias=False),
            CustomTanh()
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(2*128, 128, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 9, bias=False),
            CustomTanh()
        )
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, enc_fts):
        keys = sorted(enc_fts.keys())
        x = torch.cat(tuple(enc_fts[k] for k in keys), dim=-1)
        return {0: self.mlp0(x).view(-1, 3, 3), 1: self.mlp1(x).view(-1, 3, 3)}

class AgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.Q_net = Q_NN()
    
    def forward(self, states):
        enc_fts = {k: self.enc(v) for k,v in states.items()}
        return self.Q_net(enc_fts)

def get_s_from_o(o, device):
    c_fts = (o[7::8] - 0.4)/0.3

    b_fts = (o[i:i+7] for i in range(0, len(o), 8))
    b_fts = np.stack(tuple(b_fts))
    zero_check = (b_fts.sum(axis=1, keepdims=True) + 1)%2
    one_hot = np.concatenate((b_fts, zero_check), axis=1)
    idx_fts = one_hot.argmax(axis=1)

    return (torch.from_numpy(idx_fts).unsqueeze(0).to(device), torch.from_numpy(c_fts).unsqueeze(0).to(device))

class CustomAgent(AgentInterface):
    def __init__(self, env):
        self.name = 'bobo'
        self.env = env
        
        self.model = AgentModel()
        self.model.load_state_dict(torch.load('./soccer-bobo/best_p1_model'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.epsilon = 0.001

    def act(self, obs):
        actions = {}
        if np.random.random() < self.epsilon:
            for p_id in obs:
                actions[p_id] = self.env.action_space.sample()
            return actions
        
        states = {}
        for p_id in obs:
            states[p_id] = get_s_from_o(obs[p_id], self.device)
            
        self.model.eval()
        with torch.no_grad(): q = self.model(states)
        for p_id in q:
            actions[p_id] = np.argmax(q[p_id][0].cpu().numpy(), axis=-1)
        return actions