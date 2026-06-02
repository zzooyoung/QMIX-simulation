# env_cfg.py
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

# 센서, 지형, 로봇 자산 로드
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG 

import isaaclab.envs.mdp as mdp
# [핵심 추가] 관측 항목 설정을 위한 ObservationTermCfg 및 자산 지정을 위한 SceneEntityCfg 임포트
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg

# 1. 액션 설정 클래스 정의
@configclass
class RobotActionsCfg:
    joint_commands = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0
    )

# 2. 관측 설정 클래스 정의
@configclass
class RobotObservationsCfg:
    
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # [핵심 수정] 빈칸(pass)을 지우고, 로봇의 12개 관절 위치 정보를 실제 관측 데이터로 등록합니다.
        # 이렇게 하면 파이토치 stack 에러가 완벽하게 해결됩니다.
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos, 
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
    policy: PolicyCfg = PolicyCfg()

# 3. 메인 환경 설정 클래스
@configclass
class DefectMappingEnvCfg(ManagerBasedEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0)
    actions: RobotActionsCfg = RobotActionsCfg()
    observations: RobotObservationsCfg = RobotObservationsCfg()

    def __post_init__(self):
        # 물리 시뮬레이션 주기 설정 (200 Hz)
        self.sim.dt = 0.005  
        self.decimation = 4  # 에이전트 제어 주기는 50 Hz (0.02초)
        
        # 프로시저럴 Rough Terrain 자동 생성
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG.replace(color_scheme="random"),
        )
        
        # 로봇 추가 (네임스페이스 매핑)
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 로컬 관측을 위한 레이캐스터 (LiDAR 모사) 센서 장착
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 2.0]),
            mesh_prim_paths=["/World/ground"],
        )