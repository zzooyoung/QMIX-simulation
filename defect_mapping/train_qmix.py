# train_qmix.py
import os
os.environ["OMNI_GRAPH_EXECUTION_MODE"] = "pipeline"
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

from isaaclab.app import AppLauncher
# Mac 환경 스트리밍을 위해 8211 포트 웹서버 모드(livestream=2)로 가동
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# ───────────────────────────────────────────────────────────────
# [최종 해결] Isaac Sim 4.5.0 전용 C++ 플러그인 영구 음소거
import omni.log

try:
    omni.log.get_log().set_channel_level("omni.physx.tensors.plugin", omni.log.Level.FATAL)
except Exception:
    pass
# ───────────────────────────────────────────────────────────────

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
    # [차원 축소] 바퀴 4개 + LiDAR 363개 = 367차원 (ANYmal 때의 375보다 가벼워짐)
    agent_input_shape = 367   
    n_actions = 5             
    state_shape = 1089        
    
    agent_net = DRQNAgent(input_shape=agent_input_shape, n_actions=n_actions).to(device)
    mixer_net = QMixer(n_agents=n_agents, state_shape=state_shape).to(device)
    
    hypernet_params = list(agent_net.parameters()) + list(mixer_net.parameters())
    optimizer = optim.Adam(hypernet_params, lr=1e-4)

    print("\n[INFO] ========================================================")
    print("[INFO] 허스키(Husky) 로봇 부대 투입! 고속 매핑 테스트 가동")
    print("[INFO] ========================================================\n")

    # ───────────────────────────────────────────────────────────────
    # [추가] 메인 루프 시작 전, 데이터 기록용 CSV 파일 헤더 생성
    log_filename = "defect_mapping_results.csv"
    if not os.path.exists(log_filename):
        with open(log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward", "Coverage(%)", "Epsilon"])
    # ───────────────────────────────────────────────────────────────

    epsilon = 1.0          
    epsilon_min = 0.01
    epsilon_decay = 0.995 
    gamma = 0.99           
    
    obs, info = env.reset()
    episode = 0

    while simulation_app.is_running():
        hidden_states = [torch.zeros(env.num_envs, 64).to(device) for _ in range(n_agents)]
        episode_reward = 0
        
        grid_size = 100
        global_grid_map = np.zeros((grid_size, grid_size), dtype=bool)
        
        current_actions = [0, 0, 0] 
        action_update_steps = 10    
        
        for step in range(1000):
            if not simulation_app.is_running(): break
            
            # --- 데이터 전처리 (바퀴 4개 속도 분할) ---
            joint_all = obs["policy"]
            # 허스키는 바퀴가 4개이므로 인덱스가 4개씩 잘립니다.
            joints = [joint_all[:, 0:4], joint_all[:, 4:8], joint_all[:, 8:12]]
            
            lidar_a = env.scene["height_scanner_a"].data.ray_hits_w.view(env.num_envs, -1)
            lidar_b = env.scene["height_scanner_b"].data.ray_hits_w.view(env.num_envs, -1)
            lidars = [lidar_a, lidar_b, env.scene["height_scanner_c"].data.ray_hits_w.view(env.num_envs, -1)]
            
            raw_lidar_tensors = [
                env.scene["height_scanner_a"].data.ray_hits_w,
                env.scene["height_scanner_b"].data.ray_hits_w,
                env.scene["height_scanner_c"].data.ray_hits_w
            ]
            global_state = torch.cat(raw_lidar_tensors, dim=1).view(env.num_envs, -1)

            # --- 액션 선택 및 바퀴 속도 매핑 ---
            actions_list = []
            chosen_q_values = []
            
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
                
                # [직관적 제어] 4개 바퀴의 목표 회전 속도(Target Velocity) 텐서
                robot_joint_target = torch.zeros(4, device=device)
                wheel_speed = 100.0 # rad/s
                
                # 차동 구동 매핑: [앞왼쪽, 앞오른쪽, 뒤왼쪽, 뒤오른쪽]
                if action_idx == 0:   # 정지
                    pass 
                elif action_idx == 1: # 전진 (모든 바퀴 앞으로)
                    robot_joint_target[:] = wheel_speed
                elif action_idx == 2: # 제자리 좌회전 (왼바퀴 뒤로, 오른바퀴 앞으로)
                    robot_joint_target[0] = -wheel_speed
                    robot_joint_target[2] = -wheel_speed
                    robot_joint_target[1] = wheel_speed
                    robot_joint_target[3] = wheel_speed
                elif action_idx == 3: # 제자리 우회전 (왼바퀴 앞으로, 오른바퀴 뒤로)
                    robot_joint_target[0] = wheel_speed
                    robot_joint_target[2] = wheel_speed
                    robot_joint_target[1] = -wheel_speed
                    robot_joint_target[3] = -wheel_speed
                elif action_idx == 4: # 후진
                    robot_joint_target[:] = -wheel_speed
                    
                actions_list.append(robot_joint_target.unsqueeze(0))
            
            joint_actions = torch.zeros((env.num_envs, 12), device=device)
            
            try:
                print(f"[DEBUG] 최종 액션 텐서 값 확인: {joint_actions}")
                obs, info = env.step(joint_actions)
            except Exception as e:
                obs, info = env.reset()
                break
            
            # --- 실시간 전역 탐색 격자 지도 보상 연산 ---
            new_cells_mapped = 0
            for lidar_tensor in raw_lidar_tensors:
                pts_x = lidar_tensor[0, :, 0].cpu().numpy()
                pts_y = lidar_tensor[0, :, 1].cpu().numpy()
                
                for x, y in zip(pts_x, pts_y):
                    # ───────────────────────────────────────────────────────────────
                    # [에러 해결] 허공으로 날아가 무한대(inf)가 된 쓰레기 좌표 데이터 필터링
                    # ───────────────────────────────────────────────────────────────
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

            # --- QMIX 중앙 최적화 ---
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
        
        print(f"[QMIX 논문 훈련] 에피소드: {episode} | 탐색 면적: {episode_reward:.2f} | 맵 커버리지: {coverage_percent:.2f}% | 엡실론: {epsilon:.3f}")
        print("-" * 110)

        # ───────────────────────────────────────────────────────────────
        # [추가] 핵심 데이터만 조용히 CSV 파일에 누적 저장!
        with open(log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, episode_reward, coverage_percent, epsilon])
        # ───────────────────────────────────────────────────────────────
        
        obs, info = env.reset()

        if episode % 100 == 0:
            torch.save(agent_net.state_dict(), f"models/agent_ep_{episode}.pth")
            torch.save(mixer_net.state_dict(), f"models/mixer_ep_{episode}.pth")
            print(f"[INFO] 체크포인트 저장 완료: 에피소드 {episode}")

    env.close()

if __name__ == "__main__":
    main()