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
    # velocity_threshold: float = 0.05,
    max_dist: float = 0.044,      # 완전히 열렸을 때 (0점)
    min_dist: float = 0.03,     # 목표 닫힘 정도 (1점)
    max_vel_threshold: float = 0.5,   # 최대 허용 속도 (이보다 빠르면 보상 0)
    exp_scale: float = 1       # 지수 가파르기 (클수록 끝부분에서 점수가 확 오름)
) -> torch.Tensor:
    # 1. 에셋 및 기본 데이터 가져오기
    object_asset: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot_asset: Articulation = env.scene["robot"]
    
    # 2. 액션 매니저에서 인덱스 추출
    action_term = env.action_manager._terms.get(action_name)
    if action_term is None:
        return torch.zeros_like(object_asset.data.root_pos_w[:, 0])
    
    first_gripper_idx = action_term._joint_ids[0]
    
    # 3. 데이터 추출 (쉼표 제거 및 순서 조정)
    cube_pos_w = object_asset.data.root_pos_w[:, 0:3]
    # cube_vel_w = object_asset.data.root_lin_vel_w[:, 0:3]

    frame_idx = ee_frame_cfg.body_ids[0]
    ee_w = ee_frame.data.target_pos_w[:, frame_idx, :]
    dist = torch.norm(cube_pos_w - ee_w, dim=1)
    
    # 현재 위치 및 속도 데이터
    curr_pos = robot_asset.data.joint_pos[:, first_gripper_idx]
    gripper_vel = torch.abs(robot_asset.data.joint_vel[:, first_gripper_idx]) # 쉼표 제거 완료

    # 4. 정규화된 거리 및 지수 보상 계산
    closure_ratio = (max_dist - curr_pos) / (max_dist - min_dist)
    closure_ratio = torch.clamp(closure_ratio, min=0.0, max=1.0)

    # 디바이스 일치를 위해 closure_ratio.device 사용
    exp_scale_tensor = torch.tensor(exp_scale, device=closure_ratio.device)
    closed_reward = (torch.exp(exp_scale_tensor * closure_ratio) - 1) / (torch.exp(exp_scale_tensor) - 1)

    # 5. 조건 판별 및 속도 가중치
    is_near_mask = (dist < 0.01).float()
    # is_stable = torch.norm(cube_vel_w, dim=1) < velocity_threshold

    # 선형 속도 가중치 계산
    speed_factor = 1.0 - (gripper_vel / max_vel_threshold)
    speed_factor = torch.clamp(speed_factor, min=0.3, max=1.0)

    return is_near_mask * closed_reward * speed_factor

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

def bimanual_balance_reward(env, left_terms: list[str], right_terms: list[str]):
    # 1. 이번 스텝에서 계산된 전체 보상 텐서를 가져옵니다.
    # _step_reward는 보통 각 term의 인덱스나 이름으로 접근 가능한 구조입니다.
    
    left_total = 0.0
    for term in left_terms:
        term_idx = env.reward_manager._term_names.index(term)
        raw_reward = env.reward_manager._step_reward[:, term_idx]
        weight = env.reward_manager._term_cfgs[term_idx].weight
        left_total += raw_reward * weight

    right_total = 0.0
    for term in right_terms:
        term_idx = env.reward_manager._term_names.index(term)
        raw_reward = env.reward_manager._step_reward[:, term_idx]
        weight = env.reward_manager._term_cfgs[term_idx].weight
        right_total += raw_reward * weight

    # 2. 두 합계 중 최솟값 반환
    return torch.min(left_total, right_total)

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


