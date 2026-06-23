# env_cfg.py
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
# [지형 교체] 비정형 결함 구조물을 모사하기 위한 표면 노이즈 및 잔해물 지형 로드
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg, HfDiscreteObstaclesTerrainCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg

# 바퀴형 로봇(Jackal) 명세 유지
JACKAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/Jackal/jackal.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), 
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names=["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"],
            stiffness=0.0,
            damping=10.0,
        ),
    },
)

@configclass
class RobotActionsCfg:
    joint_commands_a = mdp.JointVelocityActionCfg(asset_name="robot_a", joint_names=[".*wheel.*"], scale=1.0)
    joint_commands_b = mdp.JointVelocityActionCfg(asset_name="robot_b", joint_names=[".*wheel.*"], scale=1.0)
    joint_commands_c = mdp.JointVelocityActionCfg(asset_name="robot_c", joint_names=[".*wheel.*"], scale=1.0)

@configclass
class RobotObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_vel_a = ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot_a", joint_names=[".*wheel.*"])})
        joint_vel_b = ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot_b", joint_names=[".*wheel.*"])})
        joint_vel_c = ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot_c", joint_names=[".*wheel.*"])})
        
    policy: PolicyCfg = PolicyCfg()

@configclass
class DefectMappingEnvCfg(ManagerBasedEnvCfg):
    # ───────────────────────────────────────────────────────────────
    # [에러 완벽 해결] 시뮬레이션 파이프라인 전체를 GPU(cuda:0)로 강제 동기화
    # ───────────────────────────────────────────────────────────────
    decimation: int = 4

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        device="cuda:0", 
        dt=0.01,
        render_interval=4,
    )
    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, 
        env_spacing=5.0, 
        replicate_physics=True
    )
    actions: RobotActionsCfg = RobotActionsCfg()
    observations: RobotObservationsCfg = RobotObservationsCfg()

    def __post_init__(self):
        # [추가] 순정 GPU 설정은 유지한 채, 물리 엔진 연산 간격(dt)만 안전하게 덮어씌웁니다.

        # 대낮 조명
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        sim_utils.spawn_light("/World/light", light_cfg)
        
        # ───────────────────────────────────────────────────────────────
        # [목표 지형 적용] 비정형 결함 구조물 모사 (부식 노면 & 산업 잔해물 융합)
        # ───────────────────────────────────────────────────────────────
        custom_defect_terrain = TerrainGeneratorCfg(
            size=(20.0, 20.0),
            border_width=0.0,
            num_rows=5,
            num_cols=5,
            sub_terrains={
                # 1. 거대한 잔해물 (LiDAR 시야를 차단하여 협동 탐색을 강제하는 벽 역할)
                "massive_debris": HfDiscreteObstaclesTerrainCfg(
                    proportion=0.5,
                    obstacle_width_range=(0.5, 2.0),
                    obstacle_height_range=(0.5, 1.5), # 최대 1.5m 높이의 거대한 콘크리트 기둥/벽 생성
                    num_obstacles=30                  # 구역당 30개의 장애물 빽빽하게 배치
                ),
                # 2. 험악하게 파인 부식 노면 (로봇의 주행을 방해하는 크레이터)
                "ruined_surface": HfRandomUniformTerrainCfg(
                    proportion=0.5,
                    noise_range=(0.0, 0.3),           # 최대 30cm 깊이의 싱크홀/파임 생성
                    noise_step=0.05
                )
            }
        )

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=custom_defect_terrain,
        )
        
        # 로봇 3대 배치 
        self.scene.robot_a = JACKAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_A")
        self.scene.robot_a.init_state.pos = (0.0, -1.5, 0.5)
        
        self.scene.robot_b = JACKAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_B")
        self.scene.robot_b.init_state.pos = (0.0, 0.0, 0.5)
        
        self.scene.robot_c = JACKAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_C")
        self.scene.robot_c.init_state.pos = (0.0, 1.5, 0.5)
        
        # 가상 LiDAR 장착
        self.scene.height_scanner_a = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_A/base_link",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            ray_alignment="yaw",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[2.0, 2.0]),
            mesh_prim_paths=["/World/ground/terrain"], 
        )
        
        self.scene.height_scanner_b = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_B/base_link",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[2.0, 2.0]),
            mesh_prim_paths=["/World/ground/terrain"], 
        )
        
        self.scene.height_scanner_c = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_C/base_link",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[2.0, 2.0]),
            mesh_prim_paths=["/World/ground/terrain"], 
        )