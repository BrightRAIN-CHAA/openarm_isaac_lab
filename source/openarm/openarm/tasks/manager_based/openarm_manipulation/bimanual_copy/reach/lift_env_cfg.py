# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.table_cfg import TABLE_USD_PATH #table usd adress

from isaaclab.managers import SceneEntityCfg
from source.openarm.openarm.tasks.manager_based.openarm_manipulation.bimanual_copy.reach import mdp

import math

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object_left: RigidObjectCfg | DeformableObjectCfg = MISSING
    object_right: RigidObjectCfg | DeformableObjectCfg = MISSING


    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.43, 0, 0], 
            rot=[0.707, 0, 0, -0.707]
        ),
        spawn=UsdFileCfg(
            usd_path = TABLE_USD_PATH,
            scale = (0.006, 0.01, 0.0032), #테이블 높이 0.3181
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    # 통합된 이름: object_pose
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,   # 아래 __post_init__에서 채울 부분
        resampling_time_range=(99999.0, 99999.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.1, 0.2),
            pos_y=(0.1, 0.2),
            pos_z=(0.5, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # 팔 액션 틀
    arm_action: mdp.JointPositionActionCfg = MISSING
    # 그리퍼 액션 틀 (이 이름이 중요합니다!)
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_obs = ObsTerm(
            func=mdp.get_bimanual_obs, # 아래에서 새로 만들 함수
        )
        generated_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"}
        )

        def __post_init__(self):
            super().__post_init__()
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_left_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
         params={
            "pose_range": {
                "x": (0, 0), 
                "y": (0, 0), 
                "z": (0, 0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_left"),
        },
    )

    reset_right_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0, 0), 
                "y": (0, 0), 
                "z": (0, 0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_right"),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    def __init__(self):
        # 1. 공통 패널티 (팔 구분 없음)
        self.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

        # 2. 양팔 대칭 보상 등록 (for문을 돌려 자동으로 left_, right_를 생성합니다)
        for side in ["left", "right"]:
            # 물체와 손 사이 거리 (Reaching)
            setattr(self, f"{side}_reaching_object", RewTerm(
                func=mdp.object_ee_distance,
                params={
                    "std": 0.1,
                    "object_cfg": SceneEntityCfg(f"object_{side}"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame", body_names=[f"openarm_{side}_ee_tcp"])
                },
                weight=3.0
            ))

            # 물체 들어올리기 (Lifting)
            setattr(self, f"{side}_lifting_object", RewTerm(
                func=mdp.object_is_lifted,
                params={
                    "target_height": 0.1,
                    "table_height": 0.349,
                    "object_cfg": SceneEntityCfg(f"object_{side}")
                },
                weight=10.0
            ))

            # 그리퍼 잡기 보상 (Grasped)
            setattr(self, f"{side}_gripper_grasped", RewTerm(
                func=mdp.gripper_is_grasped_reward,
                params={
                    "object_cfg": SceneEntityCfg(f"object_{side}"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame", body_names=[f"openarm_{side}_ee_tcp"]),
                    "action_name": f"{side}_gripper_action",
                },
                weight=5.0
            ))

            # # 목표 지점 추적 (Goal Tracking)
            # setattr(self, f"{side}_object_goal_tracking", RewTerm(
            #     func=mdp.object_goal_distance,
            #     params={
            #         "std": 0.3, 
            #         "minimal_height": 0.04, 
            #         "command_name": "object_pose",  # f"{side}_"를 제거!
            #         "object_cfg": SceneEntityCfg(f"object_{side}")
            #     },
            #     weight=16.0,
            # ))

            # # 정밀 추적 (Fine Grained Tracking)
            # setattr(self, f"{side}_object_goal_tracking_fine_grained", RewTerm(
            #     func=mdp.object_goal_distance,
            #     params={
            #         "std": 0.05, 
            #         "minimal_height": 0.04, 
            #         "command_name": "object_pose", # f"{side}_"를 제거!
            #         "object_cfg": SceneEntityCfg(f"object_{side}")
            #     },
            #     weight=5.0,
            # ))

            # 관절 속도 패널티 (Joint Velocity)
            self.joint_vel = RewTerm(
                func=mdp.joint_vel_l2,
                weight=-1e-2,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_.*"])}
            )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    left_object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("object_left")},
    )

    right_object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("object_right")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 5e6},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "joint_vel",  
            "weight": -1e-1, 
            "num_steps": 1e9
        },
    )

##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lift environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=2048, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
   
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # [중요] GPU 연산 가속 활성화 확인 (노트북은 필수)
        self.sim.device = "cuda" if "cuda" in self.sim.device else self.sim.device
