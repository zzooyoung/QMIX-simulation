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
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg

@configclass
class RobotActionsCfg:
    joint_commands_a = mdp.JointPositionActionCfg(asset_name="robot_a", joint_names=[".*"], scale=1.0)
    joint_commands_b = mdp.JointPositionActionCfg(asset_name="robot_b", joint_names=[".*"], scale=1.0)
    joint_commands_c = mdp.JointPositionActionCfg(asset_name="robot_c", joint_names=[".*"], scale=1.0)

@configclass
class RobotObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos_a = ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_a")})
        joint_pos_b = ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_b")})
        joint_pos_c = ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot_c")})
        
    policy: PolicyCfg = PolicyCfg()

@configclass
class DefectMappingEnvCfg(ManagerBasedEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0)
    actions: RobotActionsCfg = RobotActionsCfg()
    observations: RobotObservationsCfg = RobotObservationsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005  
        self.decimation = 4  
        
        # ---------------------------------------------------------------+
        # [해결 1] 글로벌 대낮 조명 강제 스폰 -> 어두운 화면 완벽 해결!
        # ---------------------------------------------------------------+
        light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        sim_utils.spawn_light("/World/light", light_cfg)
        
        # 프로시저럴 Rough Terrain 생성
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG.replace(color_scheme="random"),
        )
        
        # 로봇 3대 배치
        self.scene.robot_a = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_A")
        self.scene.robot_a.init_state.pos = (0.0, -1.5, 0.6)
        
        self.scene.robot_b = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_B")
        self.scene.robot_b.init_state.pos = (0.0, 0.0, 0.6)
        
        self.scene.robot_c = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_C")
        self.scene.robot_c.init_state.pos = (0.0, 1.5, 0.6)
        
        # 가상 LiDAR 레이캐스터 장착 (각 로봇 독립 센서)
        self.scene.height_scanner_a = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_A/base",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 2.0]),
            mesh_prim_paths=["/World/ground"],
        )
        
        self.scene.height_scanner_b = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_B/base",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 2.0]),
            mesh_prim_paths=["/World/ground"],
        )
        
        self.scene.height_scanner_c = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_C/base",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 2.0]),
            mesh_prim_paths=["/World/ground"],
        )

        # ---------------------------------------------------------------+
        # [해결 2] PhysX 에러 원천 차단: 3대 규모에서는 CPU 연산이 무조건 정답입니다.
        # ---------------------------------------------------------------+
        self.sim.physx.use_gpu = False # 지독한 GPU narrowphase 커널 런칭 에러를 완벽하게 우회합니다.