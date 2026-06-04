# train_qmix.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, livestream=2)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedEnv
from env_cfg import DefectMappingEnvCfg
from qmix_arch import DRQNAgent, QMixer

class EpisodeReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    def push(self, episode_data):
        self.buffer.append(episode_data)
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    def __len__(self):
        return len(self.buffer)

def main():
    env_cfg = DefectMappingEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    device = env.device

    n_agents = 3
    agent_input_shape = 375   
    n_actions = 5             
    state_shape = 1089        
    
    agent_net = DRQNAgent(input_shape=agent_input_shape, n_actions=n_actions).to(device)
    mixer_net = QMixer(n_agents=n_agents, state_shape=state_shape).to(device)
    
    hypernet_params = list(agent_net.parameters()) + list(mixer_net.parameters())
    optimizer = optim.Adam(hypernet_params, lr=1e-4)

    print("\n[INFO] ========================================================")
    print("[INFO] 관절 이중 중첩 버그 해결! 순수 변위량 Trot Gait 가동")
    print("[INFO] ========================================================\n")

    epsilon = 1.0          
    epsilon_min = 0.05
    epsilon_decay = 0.995 
    gamma = 0.99           
    
    obs, info = env.reset()
    episode = 0

    # 자산(Asset) 내부의 관절 인덱스 동적 추출
    joint_names = env.scene["robot_a"].joint_names
    lf_hfe = joint_names.index("LF_HFE")
    lf_kfe = joint_names.index("LF_KFE")
    lh_hfe = joint_names.index("LH_HFE")
    lh_kfe = joint_names.index("LH_KFE")
    rf_hfe = joint_names.index("RF_HFE")
    rf_kfe = joint_names.index("RF_KFE")
    rh_hfe = joint_names.index("RH_HFE")
    rh_kfe = joint_names.index("RH_KFE")

    while simulation_app.is_running():
        hidden_states = [torch.zeros(env.num_envs, 64).to(device) for _ in range(n_agents)]
        episode_reward = 0
        
        grid_size = 100
        global_grid_map = np.zeros((grid_size, grid_size), dtype=bool)
        
        current_actions = [0, 0, 0] 
        action_update_steps = 15     # 0.3초 동안 보행 방향 유지
        
        # 1000스텝(20초) 동안 여유롭게 탐색
        for step in range(1000):
            if not simulation_app.is_running(): break
            
            # --- [A. 데이터 전처리 파트] ---
            joint_all = obs["policy"]
            joints = [joint_all[:, 0:12], joint_all[:, 12:24], joint_all[:, 24:36]]
            
            lidar_a = env.scene["height_scanner_a"].data.ray_hits_w.view(env.num_envs, -1)
            lidar_b = env.scene["height_scanner_b"].data.ray_hits_w.view(env.num_envs, -1)
            lidars = [lidar_a, lidar_b, env.scene["height_scanner_c"].data.ray_hits_w.view(env.num_envs, -1)]
            
            raw_lidar_tensors = [
                env.scene["height_scanner_a"].data.ray_hits_w,
                env.scene["height_scanner_b"].data.ray_hits_w,
                env.scene["height_scanner_c"].data.ray_hits_w
            ]
            global_state = torch.cat(raw_lidar_tensors, dim=1).view(env.num_envs, -1)

            # --- [B. 액션 선택 및 정밀 Trot Gait 매핑] ---
            actions_list = []
            chosen_q_values = []
            t = env.sim.current_time  
            
            for i in range(n_agents):
                agent_input = torch.cat([joints[i], lidars[i]], dim=1)
                q_values, hidden_states[i] = agent_net(agent_input, hidden_states[i])
                hidden_states[i] = hidden_states[i].detach()
                
                if step % action_update_steps == 0:
                    if np.random.rand() < epsilon:
                        current_actions[i] = np.random.randint(0, n_actions)
                    else:
                        current_actions[i] = torch.argmax(q_values, dim=1).item()
                
                action_idx = current_actions[i]
                chosen_q_values.append(q_values[:, action_idx].unsqueeze(1))
                
                # ───────────────────────────────────────────────────────────────
                # [중요] 제로 텐서(All Zeros)로 시작하여 이중 중첩 버그 완벽 차단!
                # ───────────────────────────────────────────────────────────────
                robot_joint_target = torch.zeros(12, device=device)
                
                omega = 2 * np.pi * 2.0 * t  # 2.0 Hz 보행 진동수
                amp_hfe = 0.35               # 전진 허벅지 스윙 폭
                amp_kfe = 0.45               # 발을 들어 올리기 위한 무릎 개척 폭
                
                if action_idx == 0:   # 정지 (변위량 0 -> 정석 스탠딩 포즈 자동 유지)
                    pass 
                elif action_idx == 1: # 전진 (대각선 다리 쌍 교대 스윙)
                    robot_joint_target[lf_hfe] = np.sin(omega) * amp_hfe
                    robot_joint_target[lf_kfe] = np.cos(omega) * amp_kfe
                    robot_joint_target[rh_hfe] = np.sin(omega) * amp_hfe
                    robot_joint_target[rh_kfe] = np.cos(omega) * amp_kfe
                    
                    robot_joint_target[lh_hfe] = -np.sin(omega) * amp_hfe
                    robot_joint_target[lh_kfe] = -np.cos(omega) * amp_kfe
                    robot_joint_target[rf_hfe] = -np.sin(omega) * amp_hfe
                    robot_joint_target[rf_kfe] = -np.cos(omega) * amp_kfe
                elif action_idx == 2: # 좌회전 (차동 정상 보행 슬라이딩)
                    robot_joint_target[lf_hfe] = np.sin(omega) * amp_hfe
                    robot_joint_target[lh_hfe] = np.sin(omega) * amp_hfe
                    robot_joint_target[rf_hfe] = -np.sin(omega) * amp_hfe
                    robot_joint_target[rh_hfe] = -np.sin(omega) * amp_hfe
                elif action_idx == 3: # 우회전
                    robot_joint_target[lf_hfe] = -np.sin(omega) * amp_hfe
                    robot_joint_target[lh_hfe] = -np.sin(omega) * amp_hfe
                    robot_joint_target[rf_hfe] = np.sin(omega) * amp_hfe
                    robot_joint_target[rh_hfe] = np.sin(omega) * amp_hfe
                elif action_idx == 4: # 후진
                    robot_joint_target[lf_hfe] = -np.sin(omega) * amp_hfe
                    robot_joint_target[lf_kfe] = -np.cos(omega) * amp_kfe
                    robot_joint_target[rh_hfe] = -np.sin(omega) * amp_hfe
                    robot_joint_target[rh_kfe] = -np.cos(omega) * amp_kfe
                    
                    robot_joint_target[lh_hfe] = np.sin(omega) * amp_hfe
                    robot_joint_target[lh_kfe] = np.cos(omega) * amp_kfe
                    robot_joint_target[rf_hfe] = np.sin(omega) * amp_hfe
                    robot_joint_target[rf_kfe] = np.cos(omega) * amp_kfe
                    
                actions_list.append(robot_joint_target.unsqueeze(0))
            
            joint_actions = torch.cat(actions_list, dim=1)
            
            try:
                obs, info = env.step(joint_actions)
            except Exception as e:
                obs, info = env.reset()
                break
            
            # --- [C. 실시간 전역 탐색 격자 지도 보상 연산] ---
            new_cells_mapped = 0
            for lidar_tensor in raw_lidar_tensors:
                pts_x = lidar_tensor[0, :, 0].cpu().numpy()
                pts_y = lidar_tensor[0, :, 1].cpu().numpy()
                
                for x, y in zip(pts_x, pts_y):
                    grid_x = int((x + 10) / 0.2)
                    grid_y = int((y + 10) / 0.2)
                    
                    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                        if not global_grid_map[grid_x, grid_y]:
                            global_grid_map[grid_x, grid_y] = True
                            new_cells_mapped += 1
            
            reward_val = new_cells_mapped * 0.05
            reward = torch.tensor([[reward_val]], device=device)
            episode_reward += reward_val

            # --- [D. QMIX 중앙 최적화] ---
            agent_qs_tensor = torch.cat(chosen_q_values, dim=1) 
            q_tot = mixer_net(agent_qs_tensor, global_state)
            
            target_q_tot = reward + gamma * q_tot.detach()
            loss = F.mse_loss(q_tot, target_q_tot)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        episode += 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        coverage_percent = (global_grid_map.sum() / (grid_size * grid_size)) * 100
        
        print(f"[QMIX 논문 훈련] 에피소드: {episode} | 탐색 면적 점수(Reward): {episode_reward:.2f} | 구조물 매핑 커버리지: {coverage_percent:.2f}% | 엡실론: {epsilon:.3f} | Loss: {loss.item():.6f}")
        print("-" * 110)
        
        obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()