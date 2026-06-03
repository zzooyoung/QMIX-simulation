# train_qmix.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 1. 시뮬레이터 앱 실행 (최상단 고정 및 GUI 활성화)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Isaac Lab 및 수정한 파일들 로드
from isaaclab.envs import ManagerBasedEnv
from env_cfg import DefectMappingEnvCfg
from qmix_arch import DRQNAgent, QMixer

def main():
    # 2. 환경 및 하드웨어 설정을 인스턴스화
    env_cfg = DefectMappingEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    device = env.device

    # 3. QMIX 신경망 인스턴스화 및 GPU 전송
    n_agents = 3
    agent_input_shape = 1335  # 12 (관절) + 1323 (LiDAR 평면)
    n_actions = 12            # 관절 제어 차원
    state_shape = 3969        # 1323 * 3 좌표 플래튼
    
    agent_net = DRQNAgent(input_shape=agent_input_shape, n_actions=n_actions).to(device)
    mixer_net = QMixer(n_agents=n_agents, state_shape=state_shape).to(device)
    
    hypernet_params = list(agent_net.parameters()) + list(mixer_net.parameters())
    optimizer = optim.Adam(hypernet_params, lr=5e-4)

    print("\n[INFO] ========================================================")
    print("[INFO] QMIX 훈련 파이프라인 하드웨어 가속 바인딩 완료!")
    print("[INFO] ========================================================\n")

    # 4. 하이퍼파라미터 세팅
    epsilon = 1.0          # 탐색(Exploration) 확률
    epsilon_min = 0.05
    epsilon_decay = 0.995
    gamma = 0.99           # 할인율
    
    obs, info = env.reset()
    episode = 0

    while simulation_app.is_running():
        # 각 에이전트의 RNN 히든 상태 초기화
        hidden_states = [torch.zeros(env.num_envs, 64).to(device) for _ in range(n_agents)]
        episode_reward = 0
        
        # 1 에피소드 진행 (200 타임스텝)
        for step in range(200):
            if not simulation_app.is_running(): break
            
            # --- [A. 데이터 전처리 파트] ---
            joint_all = obs["policy"]
            joints = [joint_all[:, 0:12], joint_all[:, 12:24], joint_all[:, 24:36]]
            
            lidar_a = env.scene["height_scanner_a"].data.ray_hits_w.view(env.num_envs, -1)
            lidar_b = env.scene["height_scanner_b"].data.ray_hits_w.view(env.num_envs, -1)
            lidar_c = env.scene["height_scanner_c"].data.ray_hits_w.view(env.num_envs, -1)
            lidars = [lidar_a, lidar_b, lidar_c]
            
            global_state = torch.cat([env.scene["height_scanner_a"].data.ray_hits_w, 
                                     env.scene["height_scanner_b"].data.ray_hits_w, 
                                     env.scene["height_scanner_c"].data.ray_hits_w], dim=1).view(env.num_envs, -1)

            # --- [B. 액션 선택 파트 (Epsilon-Greedy)] ---
            actions_list = []
            chosen_q_values = []
            
            for i in range(n_agents):
                agent_input = torch.cat([joints[i], lidars[i]], dim=1)
                
                # DRQN 포워드 통과
                q_values, hidden_states[i] = agent_net(agent_input, hidden_states[i])
                
                # [그래프 중복 역전파 방지] 이전 연산 흔적 끊어내기
                hidden_states[i] = hidden_states[i].detach()
                
                if np.random.rand() < epsilon:
                    action = torch.sin(torch.ones(env.num_envs, n_actions) * env.sim.current_time).to(device)
                else:
                    action = torch.tanh(q_values) 
                
                actions_list.append(action)
                chosen_q_values.append(q_values.mean(dim=1, keepdim=True))
            
            joint_actions = torch.cat(actions_list, dim=1)
            
            # --- [C. 시뮬레이터 물리 스텝 구동] ---
            obs, info = env.step(joint_actions)
            
            # --- [D. 보상설정 및 QMIX 중앙 업데이트] ---
            reward = torch.tensor([[1.0]], device=device) 
            episode_reward += reward.item()

            agent_qs_tensor = torch.cat(chosen_q_values, dim=1) 
            q_tot = mixer_net(agent_qs_tensor, global_state)
            
            target_q_tot = reward + gamma * q_tot.detach()
            loss = F.mse_loss(q_tot, target_q_tot)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 에피소드 종료 후 통계 출력 및 엡실론 감쇠
        episode += 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"[훈련 리포트] 에피소드: {episode} | 총 협력 보상: {episode_reward:.2f} | 엡실론: {epsilon:.3f} | Loss: {loss.item():.6f}")
        
        obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()