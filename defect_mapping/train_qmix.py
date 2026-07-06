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
# Mac 환경 스트리밍을 위해 웹서버 모드 가동
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# ───────────────────────────────────────────────────────────────
# Isaac Sim 4.5.0 전용 C++ 플러그인 영구 음소거
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
    print(">>> 1. 메인 함수 진입 성공! <<<")
    env_cfg = DefectMappingEnvCfg()
    print(f">>> 2. EnvCfg 설정 완료: {type(env_cfg)} <<<")
    
    # 여기서 죽는다면 env 생성 내부의 __init__에서 터지는 겁니다.
    try:
        env = ManagerBasedEnv(cfg=env_cfg)
        print(">>> 3. 환경(ManagerBasedEnv) 생성 성공! <<<")
    except Exception as e:
        print(f">>> !!! 환경 생성 중 에러 발생: {e} !!!")
        raise e # 에러를 다시 던져서 강제로 트레이스백을 출력하게 함
    
    device = env.device
    print(f">>> 4. 장치 확인: {device} <<<")
    # ... 이후 코드 ...

    n_agents = 3
    # 바퀴 4개 + LiDAR 363개 = 367차원
    agent_input_shape = 1327   
    n_actions = 5             
    state_shape = 3969        
    
    agent_net = DRQNAgent(input_shape=agent_input_shape, n_actions=n_actions).to(device)
    mixer_net = QMixer(n_agents=n_agents, state_shape=state_shape).to(device)
    
    # 🔥 [QMIX 논문 적용 1] 타겟 네트워크(Target Network) 추가
    # 논문 스펙: 200 에피소드마다 타겟 네트워크를 업데이트하여 학습 안정성 확보
    target_agent_net = DRQNAgent(input_shape=agent_input_shape, n_actions=n_actions).to(device)
    target_mixer_net = QMixer(n_agents=n_agents, state_shape=state_shape).to(device)
    target_agent_net.load_state_dict(agent_net.state_dict())
    target_mixer_net.load_state_dict(mixer_net.state_dict())
    target_update_interval = 200
    
    # 🔥 [QMIX 논문 적용 2] Optimizer 변경 (Adam -> RMSprop)
    # 논문 스펙: lr=5e-4, RMSprop, weight decay 없음, momentum 없음
    hypernet_params = list(agent_net.parameters()) + list(mixer_net.parameters())
    optimizer = optim.RMSprop(hypernet_params, lr=5e-4, alpha=0.99, eps=1e-5)

    print("\n[INFO] ========================================================")
    print("[INFO] 허스키(Husky) 로봇 부대 투입! 고속 매핑 테스트 가동")
    print("[INFO] ========================================================\n")

    log_filename = "defect_mapping_results.csv"
    if not os.path.exists(log_filename):
        with open(log_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward", "Coverage(%)", "Epsilon"])

    # 🔥 [QMIX 논문 적용 3] Epsilon 선형 감소 (Linear Annealing)
    # 논문 스펙: 1.0에서 0.05까지 1000 에피소드에 걸쳐 "선형적"으로 감소
    epsilon = 1.0          
    epsilon_min = 0.05
    epsilon_decay_step = (1.0 - 0.05) / 1000.0  # 스텝이 아닌 마이너스(-) 감산 방식 적용
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
        
        # ====================================================================
        # 1000 스텝 에피소드 루프 (이 블록을 기존 루프와 통째로 교체하세요)
        # ====================================================================
        for step in range(1000):
            
            # --- 1. 원본 데이터 바인딩 확인 ---
            raw_lidar_tensors = [
                env.scene["height_scanner_a"].data.ray_hits_w,
                env.scene["height_scanner_b"].data.ray_hits_w,
                env.scene["height_scanner_c"].data.ray_hits_w
            ]
            
            # 레이더 텐서의 Shape 추적 (디버깅용)
            lidar_shape_a = raw_lidar_tensors[0].shape
            
            # --- 2. 데이터 전처리 및 Lidar 더미 방어선 구축 ---
            joint_all = obs["policy"]
            # 바퀴 4개의 상태를 로봇별로 분할 (joints 정의)
            joints = [joint_all[:, 0:4], joint_all[:, 4:8], joint_all[:, 8:12]]
            
            # 텐서가 정상 크기를 가질 때만 물리 연산 수행 (Empty Tensor 방지)
            if len(lidar_shape_a) > 1 and lidar_shape_a[1] > 0:
                pos_a = env.scene["height_scanner_a"].data.pos_w.unsqueeze(1)
                pos_b = env.scene["height_scanner_b"].data.pos_w.unsqueeze(1)
                pos_c = env.scene["height_scanner_c"].data.pos_w.unsqueeze(1)
                
                hits_a_local = torch.nan_to_num(raw_lidar_tensors[0] - pos_a, posinf=10.0, neginf=-10.0)
                hits_b_local = torch.nan_to_num(raw_lidar_tensors[1] - pos_b, posinf=10.0, neginf=-10.0)
                hits_c_local = torch.nan_to_num(raw_lidar_tensors[2] - pos_c, posinf=10.0, neginf=-10.0)
                
                lidar_a = hits_a_local.view(env.num_envs, -1)
                lidar_b = hits_b_local.view(env.num_envs, -1)
                lidar_c = hits_c_local.view(env.num_envs, -1)
            else:
                # 센서가 아직 안 깨어났거나 바인딩 실패 시 안전하게 0으로 채운 더미 제공
                lidar_a = torch.zeros(env.num_envs, 1323, device=device)
                lidar_b = torch.zeros(env.num_envs, 1323, device=device)
                lidar_c = torch.zeros(env.num_envs, 1323, device=device)

            lidars = [lidar_a, lidar_b, lidar_c]
            global_state = torch.cat(lidars, dim=1)

            # --- 3. QMIX 에이전트별 액션 선택 ---
            actions_list = []
            chosen_q_values = []
            
            for i in range(n_agents):
                # 이제 joints[i]가 위에서 정상 정의되었으므로 NameError가 나지 않습니다.
                agent_input = torch.cat([joints[i], lidars[i]], dim=1)
                q_values, hidden_states[i] = agent_net(agent_input, hidden_states[i])
                hidden_states[i] = hidden_states[i].detach()
                
                if step % action_update_steps == 0:
                    if np.random.rand() < epsilon:
                        # 64개 환경 각각에 대해 무작위 액션을 생성하도록 변경
                        current_actions[i] = torch.randint(0, n_actions, (env.num_envs,), device=device)
                    else:
                        # .item()을 제거하여 64개 로봇의 최고 가치 액션을 텐서 그대로 유지
                        current_actions[i] = torch.argmax(q_values, dim=1)
                
                action_idx = current_actions[i] # 이제 64차원 텐서입니다.
                chosen_q_values.append(q_values.gather(1, action_idx.unsqueeze(1))) # 안전하게 64개 환경의 Q값 수집
                
                robot_joint_target = torch.zeros((env.num_envs, 4), device=device)
                wheel_speed = 100.0
                
                # 각 환경별로 선택된 액션 인덱스에 따라 모터 속도 매핑 (인덱스 조건 분기)
                # 만약 기존에 단일 값(if action_idx == 1:) 구조로 되어 있다면 아래처럼 환경별 마스크로 처리해 주어야 64개 환경이 따로 움직입니다.
                robot_joint_target[action_idx == 1] = wheel_speed
                
                # 전진/후진/회전 매핑 구역
                robot_joint_target[action_idx == 2, 0] = -wheel_speed
                robot_joint_target[action_idx == 2, 2] = -wheel_speed
                robot_joint_target[action_idx == 2, 1] = wheel_speed
                robot_joint_target[action_idx == 2, 3] = wheel_speed
                
                robot_joint_target[action_idx == 3, 0] = wheel_speed
                robot_joint_target[action_idx == 3, 2] = wheel_speed
                robot_joint_target[action_idx == 3, 1] = -wheel_speed
                robot_joint_target[action_idx == 3, 3] = -wheel_speed
                
                robot_joint_target[action_idx == 4] = -wheel_speed
                    
                actions_list.append(robot_joint_target)
            
            joint_actions = torch.cat(actions_list, dim=1)
            
            # --- 4. 환경 스텝 전진 ---
            try:
                obs, info = env.step(joint_actions)
            except Exception as e:
                obs, info = env.reset()
                break
            
            # --- 5. 실시간 전역 탐색 격자 지도 보상 연산 ---
            # --- 실시간 전역 탐색 격자 지도 보상 연산 (로봇 상대 좌표계 맵핑으로 완벽 보완) ---
            new_cells_mapped = 0
            sample_z_hits = []

            if len(lidar_shape_a) > 1 and lidar_shape_a[1] > 0:
                # 0번 환경에 있는 로봇 A의 현재 실제 위치(원점)를 가져옵니다.
                robot_pos_w = env.scene["robot_a"].data.root_pos_w[0].cpu().numpy()
                
                for lidar_tensor in raw_lidar_tensors:
                    # 0번 환경의 레이저 센서 데이터 추출
                    pts_x = lidar_tensor[0, :, 0].cpu().numpy()
                    pts_y = lidar_tensor[0, :, 1].cpu().numpy()
                    pts_z = lidar_tensor[0, :, 2].cpu().numpy()
                    
                    if len(pts_z) > 0:
                        sample_z_hits.append(pts_z[0])
                    
                    for x, y in zip(pts_x, pts_y):
                        if not np.isfinite(x) or not np.isfinite(y) or abs(x) > 100.0 or abs(y) > 100.0:
                            continue
                        
                        # [💡 핵심 교정] 월드 절대 좌표 대신, "현재 로봇 위치 기준 상대 거리"로 격자판에 마킹합니다!
                        rel_x = x - robot_pos_w[0]
                        rel_y = y - robot_pos_w[1]
                        
                        # 로봇 반경 +-5m 범위를 100x100 격자(한 칸에 10cm)로 매핑
                        grid_x = int((rel_x + 5.0) / 0.1)
                        grid_y = int((rel_y + 5.0) / 0.1)
                        
                        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                            if not global_grid_map[grid_x, grid_y]:
                                global_grid_map[grid_x, grid_y] = True
                                new_cells_mapped += 1
                                
            # 🔥 [QMIX 논문 적용 4] 채찍과 당근(보상 및 페널티) 명시
            # 탐색을 못하고 제자리에 있으면 감점을 주어 억지로 흩어지게 만듭니다.
            if new_cells_mapped > 0:
                reward_val = new_cells_mapped * 0.05
            else:
                reward_val = -0.01  
                
            reward = torch.tensor([[reward_val]], device=device)
            episode_reward += reward_val

            # [초고속 실시간 진단 모듈] 10스텝마다 상태 중계
            if step % 10 == 0:
                current_mapped = global_grid_map.sum()
                z_status = [f"{z:.2f}m" for z in sample_z_hits] if sample_z_hits else "Empty"
                print(f"[LIVE HARDWARE DIAG] Step {step:3d} | Lidar Shape: {list(lidar_shape_a)} | 매핑된 격자: {current_mapped:4d}칸 | 타격 높이: {z_status}")

            # --- QMIX 중앙 최적화 ---
            agent_qs_tensor = torch.cat(chosen_q_values, dim=1) 
            q_tot = mixer_net(agent_qs_tensor, global_state)
            
            # 🔥 [QMIX 논문 적용 5] Target Network를 활용한 Q-value 타겟 계산
            # 기존의 q_tot.detach() 대신, 안정적인 target_mixer_net을 사용하여 학습 흔들림 방지
            with torch.no_grad():
                target_q_tot = reward + gamma * target_mixer_net(agent_qs_tensor, global_state)
                
            loss = F.mse_loss(q_tot, target_q_tot)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        episode += 1
        # 🔥 [QMIX 논문 적용 3 계속] Epsilon 선형 감소 (곱하기가 아닌 빼기)
        epsilon = max(epsilon_min, epsilon - epsilon_decay_step)
        
        coverage_percent = (global_grid_map.sum() / (grid_size * grid_size)) * 100
        
        print(f"[QMIX 논문 훈련] 에피소드: {episode} | 탐색 면적: {episode_reward:.2f} | 맵 커버리지: {coverage_percent:.2f}% | 엡실론: {epsilon:.3f}")
        print("-" * 110)

        with open(log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, f"{episode_reward:.2f}", f"{coverage_percent:.2f}", f"{epsilon:.3f}"])
        
        # 주영님의 기존 코드 부분 (에피소드 끝나는 지점)
        obs, info = env.reset()
        
        # 🔥 [QMIX 논문 적용 1 계속] 200 에피소드마다 타겟 네트워크 주기적 동기화
        if episode % target_update_interval == 0:
            target_agent_net.load_state_dict(agent_net.state_dict())
            target_mixer_net.load_state_dict(mixer_net.state_dict())
            print(f"🔄 [Target Sync] 에피소드 {episode} - 타겟 네트워크 가중치 동기화 완료!")
        
        # ───────────────────────────────────────────────────────────────
        # 🔥 [안전 종료 장치] 2000 에피소드 도달 시 자동 종료
        if episode >= 2000: 
            print("\n" + "="*50)
            print(f"🎉 QMIX 훈련 목표 에피소드({episode}회) 달성 완료! 🎉")
            print(f"최종 맵 커버리지: {coverage_percent:.2f}% | 학습을 종료합니다.")
            print("="*50 + "\n")
            break 
        # ───────────────────────────────────────────────────────────────

        if episode % 100 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(agent_net.state_dict(), f"models/agent_ep_{episode}.pth")
            torch.save(mixer_net.state_dict(), f"models/mixer_ep_{episode}.pth")
            print(f"[INFO] 체크포인트 저장 완료: 에피소드 {episode}")

    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "="*80)
        print("🚨 파이썬 실행 중 치명적 에러 발생! (엔진이 꺼지기 전에 잡았습니다) 🚨")
        print("="*80)
        import traceback
        traceback.print_exc() 
        print("="*80 + "\n")
    finally:
        # 에러를 출력한 뒤에 안전하게 엔진을 종료합니다.
        simulation_app.close()