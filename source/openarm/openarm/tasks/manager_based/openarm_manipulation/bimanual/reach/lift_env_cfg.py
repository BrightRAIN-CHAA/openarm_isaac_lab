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
from . import mdp

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
    """Command terms for the MDP."""
    
    left_object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,   # will be set by agent env cfg
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

    right_object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,   # will be set by agent env cfg
        resampling_time_range=(99999.0, 99999.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.1, 0.2),
            pos_y=(-0.2, -0.1),
            pos_z=(0.5, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # will be set by agent env cfg
    left_arm_action: (
        mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

    right_arm_action: (
        mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint1",
                                                                    "openarm_left_joint2",
                                                                    "openarm_left_joint3",
                                                                    "openarm_left_joint4",
                                                                    "openarm_left_joint5",
                                                                    "openarm_left_joint6",
                                                                    "openarm_left_joint7",
                                                                    "openarm_left_finger.*",
                                                                  ])
            },
            # noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint1",
                                                                    "openarm_right_joint2",
                                                                    "openarm_right_joint3",
                                                                    "openarm_right_joint4",
                                                                    "openarm_right_joint5",
                                                                    "openarm_right_joint6",
                                                                    "openarm_right_joint7",
                                                                    "openarm_right_finger.*",
                                                                  ])
            },
            # noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint1",
                                                                    "openarm_left_joint2",
                                                                    "openarm_left_joint3",
                                                                    "openarm_left_joint4",
                                                                    "openarm_left_joint5",
                                                                    "openarm_left_joint6",
                                                                    "openarm_left_joint7",
                                                                    "openarm_left_finger.*",
                                                                  ])
            },
            # noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint1",
                                                                    "openarm_right_joint2",
                                                                    "openarm_right_joint3",
                                                                    "openarm_right_joint4",
                                                                    "openarm_right_joint5",
                                                                    "openarm_right_joint6",
                                                                    "openarm_right_joint7",
                                                                    "openarm_right_finger.*",
                                                                  ])
            },
            # noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        left_object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_left")}
        )
        right_object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_right")}
        )

        left_target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_object_pose"}
        )
        right_target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_object_pose"}
        )

        left_actions = ObsTerm(func=mdp.last_action,
                params={
                "action_name": "left_arm_action"})

        right_actions = ObsTerm(func=mdp.last_action,
                params={
                "action_name": "right_arm_action"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
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

    left_reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.05,
            "object_cfg": SceneEntityCfg("object_left"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame", body_names=["openarm_left_ee_tcp"])
        },
        weight=2.0
    )

    right_reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.05,
            "object_cfg": SceneEntityCfg("object_right"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame", body_names=["openarm_right_ee_tcp"])
        },
        weight=2.0
    )

    left_lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={
            "target_height": 0.1, #책상으로부터 높이
            "table_height": 0.349,
            "object_cfg": SceneEntityCfg("object_left")
        },
        weight=15.0
)
    right_lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={
            "target_height": 0.1, #책상으로부터 높이
            "table_height": 0.349,
            "object_cfg": SceneEntityCfg("object_right")
        },
        weight=15.0
    )

    left_gripper_close = RewTerm(
        func=mdp.gripper_is_closed_reward,
        params={
            "object_cfg": SceneEntityCfg("object_left"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame", body_names=["openarm_left_ee_tcp"]),
            "action_name": "left_gripper_action"
        },
        weight=5.0
    )

    right_gripper_close = RewTerm(
        func=mdp.gripper_is_closed_reward,
        params={
            "object_cfg": SceneEntityCfg("object_right"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame", body_names=["openarm_right_ee_tcp"]),
            "action_name": "right_gripper_action"
        },
        weight=5.0
    )

    # left_object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "left_object_pose", "object_cfg": SceneEntityCfg("object_left")},
    #     weight=16.0,
    # )

    # right_object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "right_object_pose", "object_cfg": SceneEntityCfg("object_right")},
    #     weight=16.0,
    # )

    # left_object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "left_object_pose", "object_cfg": SceneEntityCfg("object_left")},
    #     weight=5.0,
    # )

    # right_object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "right_object_pose", "object_cfg": SceneEntityCfg("object_right")},
    #     weight=5.0,
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint1",
                                                                    "openarm_left_joint2",
                                                                    "openarm_left_joint3",
                                                                    "openarm_left_joint4",
                                                                    "openarm_left_joint5",
                                                                    "openarm_left_joint6",
                                                                    "openarm_left_joint7",
                                                                    #"openarm_left_finger.*",
                                                                  ])},
    )
    
    left_finger_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_finger.*"])},
    )

    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint1",
                                                                    "openarm_right_joint2",
                                                                    "openarm_right_joint3",
                                                                    "openarm_right_joint4",
                                                                    "openarm_right_joint5",
                                                                    "openarm_right_joint6",
                                                                    "openarm_right_joint7",
                                                                    #"openarm_right_finger.*",
                                                                  ])},
    )

    right_finger_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger.*"])},
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
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 1e9},
    )

    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -1e-2, "num_steps": 1e9},
    )

    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -1e-2, "num_steps": 1e9},
    )

    # reach_weight_left = CurrTerm(
    # func=mdp.modify_reward_weight,
    # params={
    #     "term_name": "left_reaching_object",
    #     "weight": 1.5,
    #     "num_steps": 1e9,
    #     },
    # )

    # reach_weight_right = CurrTerm(
    # func=mdp.modify_reward_weight,
    # params={
    #     "term_name": "right_reaching_object",
    #     "weight": 1.5,
    #     "num_steps": 1e9,
    #     },
    # )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lift environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
