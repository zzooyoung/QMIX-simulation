# qmix_arch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================================================================
# 1. 개별 로봇 에이전트 네트워크 (DRQN 스타일)
# =======================================================================
class DRQNAgent(nn.Module):
    def __init__(self, input_shape=1335, n_actions=12): # 관절12 + LiDAR1323 = 1335 / 출력 관절 제어 12
        super(DRQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.rnn = nn.GRUCell(64, 64) # 과거 탐색 기억 유지를 위한 RNN 레이어
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state)
        q_values = self.fc2(h)
        return q_values, h

# =======================================================================
# 2. QMIX 중앙 믹싱 네트워크 & 하이퍼네트워크
# =======================================================================
class QMixer(nn.Module):
    def __init__(self, n_agents=3, state_shape=3969, mixing_embed_dim=32): # state: 1323 * 3 좌표 = 3969
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.embed_dim = mixing_embed_dim

        # 글로벌 상태(s_t)로부터 믹싱넷의 가중치(W1)를 가상으로 생성하는 하이퍼네트워크
        self.hyper_w1 = nn.Linear(self.state_shape, self.n_agents * self.embed_dim)
        # 믹싱넷의 바이어스(b1) 생성
        self.hyper_b1 = nn.Linear(self.state_shape, self.embed_dim)

        # 최종 Q_tot를 만들기 위한 두 번째 레이어 가중치(W2) 및 바이어스(b2) 생성
        self.hyper_w2 = nn.Linear(self.state_shape, self.embed_dim * 1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_shape, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        # agent_qs 차원: [batch_size, n_agents] -> 각 로봇의 선택된 행동 가치
        # states 차원: [batch_size, state_shape] -> 플래튼된 글로벌 센서 맵 상태
        bs = agent_qs.size(0)
        states = states.view(-1, self.state_shape)

        # First layer 가중치 생성 및 단조성 만족을 위한 절대값(abs) 처리
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        
        # 믹싱넷 첫 번째 레이어 통과
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Second layer 가중치 생성 및 절대값 처리
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        # 최종 단일 팀 가치 가중치 Q_tot 산출
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(bs, -1)