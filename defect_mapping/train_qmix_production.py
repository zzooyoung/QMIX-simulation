import argparse
import os
import csv
import numpy as np

# =======================================================================
# [필수] 1. Isaac Lab 시뮬레이션 엔진 초기화 (이 부분이 최상단에 있어야 안 튕깁니다)
# =======================================================================
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train QMIX for Jackal Defect Mapping")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 물리 엔진 시동 (이 코드가 지나가야 뒤에서 에러가 안 납니다)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 엔진 시동 후 PyTorch 및 Isaac Lab 모듈 Import
import torch
import torch.nn as nn
import torch.nn.functional as F
from omni.isaac.lab.envs import ManagerBasedRLEnv

# 주의: 주영님의 환경 설정 파일(env_cfg.py) 경로에 맞게 아래 import를 수정하세요!
# 예시: from custom_env.jackal_env_cfg import JackalDefectEnvCfg
# =======================================================================

# GPU 비동기 파이프라인 충돌 방지
os.environ["OMNI_GRAPH_EXECUTION_MODE"] = "pipeline"

# =======================================================================
# 2. Jackal 전용 에이전트 네트워크 (DRQN)
# =======================================================================
class DRQNAgent(nn.Module):
    def __init__(self, input_shape=1327, n_actions=4): 
        # 바퀴 4개 + 라이다 1323 (21x21x3) = 1327
        super(DRQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.rnn = nn.GRUCell(64, 64) 
        self.fc2 = nn.Linear(64, n_actions)

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, 64)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state)
        q_values = self.fc2(h)
        return q_values, h

# =======================================================================
# 3. 고성능 서버용 QMIX 중앙 믹싱 네트워크
# =======================================================================
class QMixer(nn.Module):
    def __init__(self, n_agents=3, state_shape=3969, mixing_embed_dim=64):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.embed_dim = mixing_embed_dim

        self.hyper_w1 = nn.Linear(self.state_shape, self.n_agents * self.embed_dim)
        self.hyper_b1 = nn.Linear(self.state_shape, self.embed_dim)
        self.hyper_w2 = nn.Linear(self.state_shape, self.embed_dim * 1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_shape, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.view(-1, self.state_shape)

        w1 = torch.abs(self.hyper_w1(states)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        w2 = torch.abs(self.hyper_w2(states)).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(bs, -1)

# =======================================================================
# 4. 메인 훈련 파이프라인
# =======================================================================


def build_marl_tensors(obs, env):
    """
    Isaac Lab의 기본 관측값(바퀴 속도)과 LiDAR 센서값을 추출하여
    DRQN 에이전트 입력 차원(1327)과 QMIX 글로벌 상태(3969)로 조립합니다.
    """
    bs = obs.shape[0] # 배치 사이즈 (num_envs)
    
    # 1. 바퀴 속도 분리 (env_cfg.py의 observations 설정에 따라 12차원이 들어옵니다)
    # [num_envs, 12] -> 각각 [num_envs, 4]
    wheel_a = obs[:, 0:4]
    wheel_b = obs[:, 4:8]
    wheel_c = obs[:, 8:12]

    # 2. 물리 엔진 씬(Scene)에서 직접 LiDAR 센서 데이터 추출
    scene = env.unwrapped.scene
    hits_a = scene["height_scanner_a"].data.ray_hits_w
    hits_b = scene["height_scanner_b"].data.ray_hits_w
    hits_c = scene["height_scanner_c"].data.ray_hits_w
    
    # [핵심] 월드 좌표(World Coords)를 로봇 기준 상대 좌표(Local Coords)로 변환!
    # NN은 절대 좌표를 주면 학습하지 못합니다. "내 로봇에서 얼마나 떨어져 있는가?"로 바꿔줍니다.
    orig_a = scene["height_scanner_a"].data.pos_w.unsqueeze(1)
    orig_b = scene["height_scanner_b"].data.pos_w.unsqueeze(1)
    orig_c = scene["height_scanner_c"].data.pos_w.unsqueeze(1)

    # 차원 펴기(Flatten): [num_envs, 441, 3] -> [num_envs, 1323]
    lidar_local_a = (hits_a - orig_a).view(bs, -1)
    lidar_local_b = (hits_b - orig_b).view(bs, -1)
    lidar_local_c = (hits_c - orig_c).view(bs, -1)

    # 3. 개별 에이전트 입력 텐서 조립 (바퀴 4 + 라이다 1323 = 1327차원)
    agent_in_a = torch.cat([wheel_a, lidar_local_a], dim=1)
    agent_in_b = torch.cat([wheel_b, lidar_local_b], dim=1)
    agent_in_c = torch.cat([wheel_c, lidar_local_c], dim=1)

    # 4. QMIX 글로벌 상태 조립 (세 로봇의 LiDAR 지도를 하나로 합침 = 3969차원)
    global_state = torch.cat([lidar_local_a, lidar_local_b, lidar_local_c], dim=1)

    # 반환값: [에이전트별 입력 리스트], 글로벌 상태, [맵핑용 원본 월드 좌표 LiDAR]
    return [agent_in_a, agent_in_b, agent_in_c], global_state, [hits_a, hits_b, hits_c]

def train_production_pipeline(env, device="cuda"):
    print("[INFO] 본격적인 QMIX 훈련 파이프라인을 시작합니다!")
    n_agents = 3
    gamma = 0.99
    epsilon = 0.995
    epsilon_decay = 0.992  
    epsilon_min = 0.02
    wheel_speed_scale = 100.0  
    grid_size = 100
    
    agent_net = DRQNAgent().to(device)
    mixer_net = QMixer().to(device)
    optimizer = torch.optim.Adam(
        list(agent_net.parameters()) + list(mixer_net.parameters()), lr=1e-4
    )
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_filename = os.path.join(current_dir, "defect_mapping_results.csv")
    os.makedirs(os.path.join(current_dir, "models"), exist_ok=True)
    
    if not os.path.exists(log_filename):
        with open(log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward", "Coverage(%)", "Epsilon"])

    episode = 0

    while True: 
        obs, info = env.reset()
        agent_inputs, global_state, raw_lidar_tensors = build_marl_tensors(obs, env) # << 여기 추가!
        global_grid_map = np.zeros((grid_size, grid_size), dtype=bool)
        episode_reward = 0.0
        
        hidden_states = [agent_net.init_hidden().to(device) for _ in range(n_agents)]
        
        while True:
            chosen_q_values = []
            joint_actions_list = []
            
            for i in range(n_agents):
                # 텐서 슬라이싱
                agent_input = agent_inputs[i]
                q_values, hidden_states[i] = agent_net(agent_input, hidden_states[i])
                
                if np.random.rand() < epsilon:
                    action_idx = np.random.randint(0, 4)
                else:
                    action_idx = torch.argmax(q_values, dim=1).item()
                
                chosen_q_values.append(q_values[:, action_idx].unsqueeze(1))
                
                if action_idx == 0:   act = [1.0, 1.0, 1.0, 1.0]     
                elif action_idx == 1: act = [-1.0, -1.0, -1.0, -1.0] 
                elif action_idx == 2: act = [-0.5, 0.5, -0.5, 0.5]   
                else:                 act = [0.5, -0.5, 0.5, -0.5]   
                
                joint_actions_list.extend([a * wheel_speed_scale for a in act])
            
            joint_actions = torch.tensor([joint_actions_list], device=device)
            
            try:
                obs, rewards, terminated, truncated, info = env.step(joint_actions)
                agent_inputs, global_state, raw_lidar_tensors = build_marl_tensors(obs, env)
            except Exception as e:
                print(f"[SYSTEM ERROR] 환경 런타임 예외 발생: {e}")
                break
                
            new_cells_mapped = 0
            raw_lidar_tensors = info.get("raw_lidar", [])
            
            for lidar_tensor in raw_lidar_tensors:
                pts_x = lidar_tensor[0, :, 0].cpu().numpy()
                pts_y = lidar_tensor[0, :, 1].cpu().numpy()
                
                for x, y in zip(pts_x, pts_y):
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    
                    grid_x = int((x + 10) / 0.2)
                    grid_y = int((y + 10) / 0.2)
                    
                    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                        if not global_grid_map[grid_x, grid_y]:
                            global_grid_map[grid_x, grid_y] = True
                            new_cells_mapped += 1
            
            reward_val = new_cells_mapped * 0.05
            reward = torch.tensor([[reward_val]], device=device)
            episode_reward += reward_val
            
            agent_qs_tensor = torch.cat(chosen_q_values, dim=1) 
            global_state = info.get("global_state", torch.zeros(1, 3969, device=device))
            
            q_tot = mixer_net(agent_qs_tensor, global_state)
            target_q_tot = reward + gamma * q_tot.detach()
            loss = F.mse_loss(q_tot, target_q_tot)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if terminated or truncated:
                break
        
        episode += 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        coverage_percent = (global_grid_map.sum() / (grid_size * grid_size)) * 100
        
        with open(log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, f"{episode_reward:.2f}", f"{coverage_percent:.2f}", f"{epsilon:.3f}"])
            
        print(f"[QMIX 논문 훈련] 에피소드: {episode} | 탐색 면적: {episode_reward:.2f} | 맵 커버리지: {coverage_percent:.2f}% | 엡실론: {epsilon:.3f}")
        
        if episode % 100 == 0:
            torch.save(agent_net.state_dict(), os.path.join(current_dir, f"models/agent_ep_{episode}.pth"))
            torch.save(mixer_net.state_dict(), os.path.join(current_dir, f"models/mixer_ep_{episode}.pth"))
            print(f"[INFO] {episode} 에피소드 체크포인트 저장 완료.")

# =======================================================================
# 5. 실행 엔트리 포인트 (환경 생성 및 파이프라인 호출)
# =======================================================================
if __name__ == "__main__":
    try:
        # 주영님이 작성하신 환경 설정 파일(env_cfg.py)에서 
        # DefectMappingEnvCfg 클래스를 가져옵니다.
        # (만약 파일 이름이나 경로가 다르다면 'from env_cfg' 부분을 실제 경로로 맞춰주세요)
        from env_cfg import DefectMappingEnvCfg
        
        print(">>> Isaac Lab 환경 설정을 불러옵니다... <<<")
        env_cfg = DefectMappingEnvCfg() 
        env = ManagerBasedRLEnv(cfg=env_cfg)
        
        print(">>> Isaac Lab 물리 엔진 준비 완료. 본격적인 학습을 시작합니다! <<<")
        
        # 메인 훈련 파이프라인 호출 (관측 텐서 변환 로직이 포함된 파이프라인)
        train_production_pipeline(env)
        
    except Exception as e:
        print(f"[SYSTEM ERROR] 실행 중 치명적 에러 발생: {e}")
        # 에러가 발생한 정확한 줄 번호와 원인을 터미널에 상세히 출력합니다.
        import traceback
        traceback.print_exc() 
    finally:
        # 스크립트가 끝나면 엔진을 안전하게 종료하여 좀비 프로세스가 남지 않도록 합니다.
        if 'simulation_app' in locals():
            simulation_app.close()