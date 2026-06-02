# test_agent.py
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedEnv
from env_cfg import DefectMappingEnvCfg

def main():
    env_cfg = DefectMappingEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    
    print("\n[INFO] ========================================================")
    print(f"[INFO] Isaac Lab 2.x 기반 QMIX 멀티 로봇 환경(에이전트 3대) 구축 성공!")
    print(f"[INFO] 전체 액션 입력 벡터 제어 차원: {env.action_manager.total_action_dim} 차원")
    print("[INFO] ========================================================\n")
    
    env.reset()
    while simulation_app.is_running():
        random_actions = torch.sin(
            torch.ones(env.num_envs, env.action_manager.total_action_dim) * env.sim.current_time
        ).to(env.device)
        
        obs, info = env.step(random_actions)
        
        # ---------------------------------------------------------------+
        # [핵심 수정] pos_w 대신 ray_hits_w를 사용하여 진짜 가상 LiDAR 충돌점 추출
        # ---------------------------------------------------------------+
        lidar_a = env.scene["height_scanner_a"].data.ray_hits_w
        lidar_b = env.scene["height_scanner_b"].data.ray_hits_w
        lidar_c = env.scene["height_scanner_c"].data.ray_hits_w
        
        # QMIX 중앙 집중형 가치 함수(Mixing Net)용 글로벌 상태(s_t) 조립
        global_state_space = torch.cat([lidar_a, lidar_b, lidar_c], dim=1)
        
        if env.sim.current_time % 1.0 < 0.02: 
            print(f"[디버그] 현재 시뮬레이션 시간: {env.sim.current_time:.2f}s")
            print(f" └─ 개별 옵저베이션 벡터 형태 (obs['policy'] Shape) : {obs['policy'].shape}")
            print(f" └─ 로봇A LiDAR 스캔 데이터 차원 형태 (Shape) : {lidar_a.shape}")
            print(f" └─ QMIX 중앙 믹싱용 글로벌 텐서 결합 차원 (s_t Shape) : {global_state_space.shape}")
            print("-" * 80)

    env.close()

if __name__ == "__main__":
    main()