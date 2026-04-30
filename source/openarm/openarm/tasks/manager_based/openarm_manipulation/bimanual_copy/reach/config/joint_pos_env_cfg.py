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

import math

from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg as EventTerm

from source.openarm.openarm.tasks.manager_based.openarm_manipulation.bimanual_copy.reach import mdp
from source.openarm.openarm.tasks.manager_based.openarm_manipulation.bimanual_copy.reach.lift_env_cfg import LiftEnvCfg

from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)
from isaaclab.assets.articulation import ArticulationCfg


from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import SceneEntityCfg

from isaaclab.markers.config import FRAME_MARKER_CFG
##
# Environment configuration
##

@configclass
class OpenArmCubeLiftEnvCfg(LiftEnvCfg):

    def __post_init__(self):
        # 부모 클래스 초기화
        super().__post_init__()

        # --- [노트북 최적화] 환경 개수 조정 ---
        # 지능 공유를 쓰면 데이터가 2배가 되므로, 환경은 2048개만 열어도 충분합니다.
        self.scene.num_envs = 2048 
        self.scene.env_spacing = 2.5

        # --- [로봇 설정] OpenArm 초기 포즈 ---
        # 반복되는 조인트 설정을 간략화했습니다.
        joint_init_pos = {
            f"openarm_{side}_joint{i}": (2.0 if i == 4 else 0.0) 
            for side in ["left", "right"] for i in range(1, 8)
        }
        joint_init_pos.update({f"openarm_{side}_finger_joint.*": 0.0 for side in ["left", "right"]})

        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=joint_init_pos),
        )

        # --- [액션 통합] 지능 공유의 핵심 ---
        # left/right를 따로 두지 않고 'arm_action' 하나로 합칩니다.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_joint[1-7]", "openarm_right_joint[1-7]"],
            scale=0.5,
            use_default_offset=True,
        )

        # 그리퍼 액션도 하나로 통합합니다.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*", "openarm_right_finger_joint.*"],
            open_command_expr={
                "openarm_left_finger_joint.*": 0.044,
                "openarm_right_finger_joint.*": 0.044
            },
            close_command_expr={
                "openarm_left_finger_joint.*": 0.025,
                "openarm_right_finger_joint.*": 0.025
            },
        )
        # --- [물체 및 명령] 통합 버전 ---
        self.commands.object_pose.body_name = ["openarm_left_ee_tcp", "openarm_right_ee_tcp"]
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        for side in ["left", "right"]:
            y_pos = 0.3 if side == "left" else -0.3
            setattr(self.scene, f"object_{side}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Object{side.capitalize()}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, y_pos, 0.35], rot=[1, 0, 0, 0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                    scale=(0.8, 0.8, 0.8),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        disable_gravity=False,
                    ),
                ),
            ))

        # --- [시각화] Frame Transformer ---
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/openarm_{side}_ee_tcp",
                    name=f"openarm_{side}_ee_tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, -0.035)),
                ) for side in ["left", "right"]
            ],
        )

@configclass
class OpenArmCubeLiftEnvCfg_PLAY(OpenArmCubeLiftEnvCfg):
    def __post_init__(self):
        # 1. 위에서 만든 지능 공유용 설정을 그대로 가져옵니다.
        super().__post_init__()

        # 2. 테스트용이므로 환경 개수는 50개면 충분합니다. (노트북 사양 고려)
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # 3. 테스트할 때는 노이즈나 무작위 요소를 꺼서 
        # 로봇이 학습한 대로 똑바로 움직이는지 확인합니다.
        self.observations.policy.enable_corruption = False
