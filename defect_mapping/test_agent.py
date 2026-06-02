# test_agent.py
from isaaclab.app import AppLauncher

# 1. 시뮬레이터 앱 실행 (최상단 고정)
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedEnv
from env_cfg import DefectMappingEnvCfg

def main():
    # 2. 환경 설정 주입 및 인스턴스화
    env_cfg = DefectMappingEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)
    
    print("\n[INFO] =================================================")
    print("[INFO] Isaac Lab 2.x 표준 단일 로봇 핵심 환경 구축 완료!")
    print("[INFO] =================================================\n")
    
    # 3. 환경 리셋 및 메인 루프 돌리기
    env.reset()
    while simulation_app.is_running():
        # [핵심 수정] action_term_dim(리스트) 대신 total_action_dim(정수)을 사용하여 파이토치 언팩 에러를 해결합니다.
        # RTX 5080 GPU 디바이스 환경에 맞게 액션 텐서 전송
        random_actions = torch.sin(
            torch.ones(env.num_envs, env.action_manager.total_action_dim) * env.sim.current_time
        ).to(env.device)
        
        # 물리 엔진 1스텝 구동
        obs, info = env.step(random_actions)
        
        # 4. 레이캐스터 센서(가상 LiDAR)로부터 데이터 텐서 추출
        lidar_data = env.scene["height_scanner"].data.pos_w
        
        # 1초마다 터미널에 데이터 구조 출력
        if env.sim.current_time % 1.0 < 0.02: 
            print(f"[디버그] 현재 시간: {env.sim.current_time:.2f}s | LiDAR 데이터 텐서 형태(Shape): {lidar_data.shape}")

    # 루프 종료 시 앱 닫기
    env.close()

if __name__ == "__main__":
    main()