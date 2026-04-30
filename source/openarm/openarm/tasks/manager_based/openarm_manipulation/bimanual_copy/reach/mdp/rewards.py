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

def gripper_is_grasped_reward(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg, 
    ee_frame_cfg: SceneEntityCfg,
    action_name: str,
    velocity_threshold: float = 0.05,
    # 큐브(0.06)의 절반인 0.03보다 아주 살짝 큰 값으로 설정
    single_gripper_threshold: float = 0.0305
) -> torch.Tensor:
    # 1. 에셋 데이터 가져오기
    object_asset: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot_asset: Articulation = env.scene["robot"]
    
    # 2. 물체 위치 및 속도
    cube_pos_w = object_asset.data.root_pos_w[:, 0:3]
    cube_vel_w = object_asset.data.root_lin_vel_w[:, 0:3]
    
    # 3. End-Effector 위치 및 거리 계산
    frame_idx = ee_frame_cfg.body_ids[0]
    ee_w = ee_frame.data.target_pos_w[:, frame_idx, :]
    dist = torch.norm(cube_pos_w - ee_w, dim=1)
    
    # 4. 그리퍼 한쪽 관절 값 추출
    action_term = env.action_manager._terms.get(action_name)
    if action_term is None:
        return torch.zeros_like(dist)
    
    # 여러 개의 그리퍼 관절 중 첫 번째 인덱스 하나만 사용 (어차피 대칭이므로)
    first_gripper_idx = action_term._joint_ids[0]
    # 부호가 마이너스일 수 있으므로 절대값을 취해줍니다.
    single_gripper_pos = torch.abs(robot_asset.data.joint_pos[:, first_gripper_idx])

    # 5. 조건 판별 (한쪽 관절 기준)
    is_near = dist < 0.01 

    # 한쪽 관절이 0.0305보다 작으면 (즉, 안쪽으로 충분히 들어왔으면) 닫힌 것으로 간주
    is_closed = single_gripper_pos < single_gripper_threshold
    
    # 물체가 튕겨 나가지 않고 정지해 있는지 확인
    is_stable = torch.norm(cube_vel_w, dim=1) < velocity_threshold
    
    return (is_near & is_closed & is_stable).float()
    
# def gripper_is_closed_reward(
#     env: ManagerBasedRLEnv, 
#     object_cfg: SceneEntityCfg, 
#     ee_frame_cfg: SceneEntityCfg,
#     action_name: str
# ) -> torch.Tensor:
#     object_asset: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     robot_asset: Articulation = env.scene["robot"]
#     cube_pos_w = object_asset.data.root_pos_w[:, 0:3]
#     frame_idx = ee_frame_cfg.body_ids[0]
#     ee_w = ee_frame.data.target_pos_w[:, frame_idx, :]
#     dist = torch.norm(cube_pos_w - ee_w, dim=1)
#     action_term = env.action_manager._terms.get(action_name)
#     if action_term is None:
#         return torch.zeros_like(dist)
        
#     action_indices = action_term._joint_ids
    
#     gripper_pos = torch.mean(robot_asset.data.joint_pos[:, action_indices], dim=1)

#     is_near = dist < 0.02
#     is_closed = gripper_pos < 0.031
    
#     return (is_near & is_closed).float()

def object_is_lifted(
    env: ManagerBasedRLEnv,
    target_height: float, #책상으로부터 들어올릴 높이
    table_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object_height = object.data.root_pos_w[:, 2]
    lift_height = torch.clamp(object_height - table_height - 0.03, min=0.0) #최소값은 0으로 지정, 큐브 중심 위치 0.03 offset

    scale = 1.5 / target_height
    reward = torch.clamp( 1.11 * torch.tanh(scale * lift_height), min =0.0, max = 1.0) #1.11 * tanh(1.5) ≒ 1.0이 되도록
    
    return reward

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


