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
            joint_names_expr=[
                "front_left_wheel_joint", 
                "front_right_wheel_joint", 
                "rear_left_wheel_joint", 
                "rear_right_wheel_joint"
            ],
            stiffness=0.0,
            damping=0.5,
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
        env_spacing=25.0, 
        replicate_physics=True
    )
    actions: RobotActionsCfg = RobotActionsCfg()
    observations: RobotObservationsCfg = RobotObservationsCfg()
    debug_vis = None

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
                # 1. 완전한 평지 (나머지 50% 공간을 평탄하게 만들어 구덩이 원천 차단)
                "flat_ground": HfRandomUniformTerrainCfg(
                    proportion=0.5,
                    noise_range=(0.0, 0.0), # 노이즈를 0으로 고정하여 아스팔트처럼 평탄하게 만듭니다.
                    noise_step=0.1,
                ),
                # 2. 거대한 잔해물 (위로 솟아오른 벽/장애물만 생성)
                "massive_debris": HfDiscreteObstaclesTerrainCfg(
                    proportion=0.5,
                    obstacle_width_range=(0.5, 2.0),
                    obstacle_height_range=(0.5, 1.5), # 아래로 파이지 않고 위로만 최대 1.5m 솟아오릅니다.
                    num_obstacles=30                  
                ),
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
            offset=RayCasterCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.5), 
                rot=(0.7071, 0.0, -0.7061, 0.0)  # 땅을 봄 🔍 (마이너스 기호 추가)
            ),
            ray_alignment="yaw",
            attach_yaw_only=False,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[4.0, 4.0]),
            mesh_prim_paths=["/World/ground/terrain"], 
            debug_vis=False,
        )
        
        self.scene.height_scanner_b = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_B/base_link",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.5), 
                rot=(0.7071, 0.0, -0.7061, 0.0) # 땅을 봄 🔍 (마이너스 기호 추가)
            ),
            ray_alignment="yaw",
            attach_yaw_only=False,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[4.0, 4.0]),
            mesh_prim_paths=["/World/ground/terrain"], 
            debug_vis=False,
            visualizer_cfg=None,
        )
        
        self.scene.height_scanner_c = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_C/base_link",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.5), 
                rot=(0.7071, 0.0, -0.7061, 0.0)  # 땅을 봄 🔍 (마이너스 기호 추가)
            ),
            ray_alignment="yaw",
            attach_yaw_only=False,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[4.0, 4.0]),
            mesh_prim_paths=["/World/ground/terrain"], 
            debug_vis=False,
            visualizer_cfg=None,
        )