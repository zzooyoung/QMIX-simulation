# env_cfg.py
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
# [임포트 오류 해결] SubTerrainCfg 대신 정석 높이맵 파도 지형 클래스인 HfWaveTerrainCfg 로드
from isaaclab.terrains.height_field import HfWaveTerrainCfg 
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
        
        # 돔 라이트 배치 (전체를 대낮처럼 환하게)
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        sim_utils.spawn_light("/World/light", light_cfg)
        
        # ───────────────────────────────────────────────────────────────
        # [정상화 완료] 로봇이 뒤집어지지 않고 부드럽게 넘나들 수 있는 구릉지 제너레이터
        # ───────────────────────────────────────────────────────────────
        custom_gentle_terrain = TerrainGeneratorCfg(
            size=(20.0, 20.0),
            border_width=0.0,
            num_rows=5,
            num_cols=5,
            sub_terrains={
                "gentle_wave": HfWaveTerrainCfg(
                    proportion=1.0,
                    amplitude_range=(0.05, 0.15), # [API 매핑] 5cm에서 최대 15cm까지의 안전한 굴곡 높이
                    num_waves=2
                )
            }
        )

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=custom_gentle_terrain,
        )
        
        # env_cfg.py 수정 파트 (로봇 스폰 위치 Z축을 1.2로 상향 조정)
        # 로봇 3대 배치 (스폰 높이를 안전하게 1.2m로 지정하여 땅속 스폰 방지)
        self.scene.robot_a = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_A")
        self.scene.robot_a.init_state.pos = (0.0, -1.5, 1.2) # 0.6 -> 1.2로 변경
        
        self.scene.robot_b = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_B")
        self.scene.robot_b.init_state.pos = (0.0, 0.0, 1.2)  # 0.6 -> 1.2로 변경
        
        self.scene.robot_c = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_C")
        self.scene.robot_c.init_state.pos = (0.0, 1.5, 1.2)  # 0.6 -> 1.2로 변경
        
        # 가상 LiDAR 레이캐스터 장착 (제너레이터 모드이므로 /terrain 경로 타겟팅)
        self.scene.height_scanner_a = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_A/base",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[2.0, 2.0]),
            mesh_prim_paths=["/World/ground/terrain"], 
        )
        
        self.scene.height_scanner_b = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_B/base",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[2.0, 2.0]),
            mesh_prim_paths=["/World/ground/terrain"], 
        )
        
        self.scene.height_scanner_c = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_C/base",
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[2.0, 2.0]),
            mesh_prim_paths=["/World/ground/terrain"], 
        )

        # PhysX GPU 버퍼 최적화 세팅
        self.sim.physx.use_gpu = True
        self.sim.physx.gpu_max_rigid_contact_count = 2**22     
        self.sim.physx.gpu_max_rigid_patch_count = 2**20       
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**20   
        self.sim.physx.gpu_heap_capacity = 2**28               
        self.sim.physx.gpu_max_vertex_count = 2**22