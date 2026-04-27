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

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# def object_is_lifted(
#     env: ManagerBasedRLEnv,
#     minimal_height: float,
#     object_cfg: SceneEntityCfg,
# ) -> torch.Tensor:
#     """Reward the agent for lifting the object above the minimal height."""
#     object: RigidObject = env.scene[object_cfg.name]
#     return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    table_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """물체를 들어올린 높이에 비례하여 선형적으로 보상."""
    object: RigidObject = env.scene[object_cfg.name]
    current_height = object.data.root_pos_w[:, 2]
    lift_dist = current_height - table_height
    target_lift_dist = minimal_height - table_height
    reward = lift_dist / target_lift_dist
    return torch.clamp(reward, min=0.0, max=1.0)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """"Reward the agent for reaching the object using tanh-kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    frame_idx = ee_frame_cfg.body_ids[0]
    ee_w = ee_frame.data.target_pos_w[:, frame_idx, :]
    
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1) #두 위치 벡터의 크기 구하기
    return 1.0 / (1.0 + torch.square(object_ee_distance / std)) # 코시분포로 계산된 거리

    # return 1 - torch.tanh(object_ee_distance / std) - 0.3*object_ee_distance
    # return 1 - torch.tanh(object_ee_distance / std) #1 - tanh(d/std)
    
    
def gripper_is_closed_reward(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg, 
    ee_frame_cfg: SceneEntityCfg,
    action_name: str
) -> torch.Tensor:
    """물체 근처에서 그리퍼가 닫혔을 때 보상을 줍니다."""
    
    # 1. 컴포넌트 가져오기
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # 2. 위치 데이터 추출 및 거리 계산
    cube_pos_w = object.data.root_pos_w[:, 0:3]
    frame_idx = ee_frame_cfg.body_ids[0]
    ee_w = ee_frame.data.target_pos_w[:, frame_idx, :]
    dist = torch.norm(cube_pos_w - ee_w, dim=1)
    
    # 3. 그리퍼 상태 확인 (에러 발생 지점 수정)
    # action_manager에서 해당 액션이 제어하는 조인트 인덱스를 가져옵니다.
    # 만약 get_term_indices가 없다면 ._term_indices[action_name]를 시도하세요.
    try:
        action_indices = env.action_manager.get_term_indices(action_name)
    except AttributeError:
        action_indices = env.action_manager._term_indices[action_name]
        
    gripper_pos = torch.mean(env.scene["robot"].data.joint_pos[:, action_indices], dim=1)
    
    # 4. 조건부 보상 (2cm 이내 + 0.03m 보다 작게 닫힘)
    is_near = dist < 0.02
    is_closed = gripper_pos < 0.03 
    
    return (is_near & is_closed).float()

# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     object_cfg: SceneEntityCfg,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: Articulation = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(
#         robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
#     )
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
#     # rewarded if the object is lifted above the threshold
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (
#         1 - torch.tanh(distance / std)
#     )


