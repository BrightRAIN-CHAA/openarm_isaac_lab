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

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _contact_force_magnitudes(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Return contact force magnitudes for each sensor body."""
    try:
        sensor = env.scene[sensor_name]
    except KeyError:
        return torch.zeros(env.num_envs, 0, device=env.device)

    force_matrix_w = getattr(sensor.data, "force_matrix_w", None)
    if force_matrix_w is not None:
        # Shape: (num_envs, num_sensor_bodies, num_filtered_bodies, 3).
        force_mag = torch.norm(force_matrix_w, dim=-1)
        if force_mag.ndim > 2:
            force_mag = torch.max(force_mag, dim=-1).values
        return force_mag

    # Fallback for unfiltered sensors. Shape: (num_envs, num_sensor_bodies, 3).
    return torch.norm(sensor.data.net_forces_w, dim=-1)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    broad_distance: float = 0.2,
    near_distance: float = 0.01,
    slope_ratio: float = 4.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """"Reward the agent for reaching the object using an exponential distance kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    frame_idx = ee_frame_cfg.body_ids[0]
    ee_w = ee_frame.data.target_pos_w[:, frame_idx, :]
    
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1) #두 위치 벡터의 크기 구하기
    slope_ratio_tensor = torch.tensor(slope_ratio, device=object_ee_distance.device)
    decay = (broad_distance - near_distance) / torch.log(slope_ratio_tensor)
    reward = torch.exp(-(object_ee_distance - near_distance) / decay)
    return torch.clamp(reward, max=1.0)

# def near_object_ee_velocity_l2(
#     env: ManagerBasedRLEnv,
#     std: float,
#     vel_scale: float = 0.2,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Penalize fast end-effector motion more strongly near the object."""
#     object: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     robot: Articulation = env.scene[robot_cfg.name]

#     frame_idx = ee_frame_cfg.body_ids[0]
#     ee_pos_w = ee_frame.data.target_pos_w[:, frame_idx, :]
#     object_pos_w = object.data.root_pos_w[:, 0:3]
#     dist = torch.norm(object_pos_w - ee_pos_w, dim=1)

#     body_idx = robot_cfg.body_ids[0]
#     ee_lin_vel_w = robot.data.body_link_lin_vel_w[:, body_idx, :]
#     ee_speed = torch.norm(ee_lin_vel_w, dim=1)
#     ee_vel_l2 = torch.square(ee_speed / vel_scale)
#     near_weight = torch.exp(-dist / std)

#     return near_weight * ee_vel_l2

def gripper_is_grasped_reward(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg, 
    ee_frame_cfg: SceneEntityCfg,
    action_name: str,
    contact_force_threshold: float = 20.0,
    contact_slope_ratio: float = 2.0,
    near_distance_threshold: float = 0.02,
) -> torch.Tensor:
    object_asset: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object_asset.data.root_pos_w[:, 0:3]

    frame_idx = ee_frame_cfg.body_ids[0]
    ee_w = ee_frame.data.target_pos_w[:, frame_idx, :]
    dist = torch.norm(cube_pos_w - ee_w, dim=1)

    near_reward = (dist < near_distance_threshold).float()

    sensor_names = (
        ("left_left_finger_contact", "left_right_finger_contact")
        if "left" in action_name
        else ("right_left_finger_contact", "right_right_finger_contact")
    )
    contact_force_values = []
    for sensor_name in sensor_names:
        contact_force_mag = _contact_force_magnitudes(env, sensor_name)
        if contact_force_mag.shape[1] > 0:
            contact_force_values.append(torch.max(contact_force_mag, dim=1).values)

    if contact_force_values:
        contact_force = torch.max(torch.stack(contact_force_values, dim=1), dim=1).values
        contact_ratio = torch.clamp(contact_force / contact_force_threshold, min=0.0, max=1.0)
        contact_exp_scale = torch.log(torch.tensor(contact_slope_ratio, device=env.device))
        contact_reward = (torch.exp(contact_exp_scale * contact_ratio) - 1.0) / (
            torch.exp(contact_exp_scale) - 1.0
        )
    else:
        contact_reward = torch.zeros(env.num_envs, device=env.device)

    return contact_reward * near_reward

def action_terms_rate_l2(
    env: ManagerBasedRLEnv,
    action_names: tuple[str, ...],
) -> torch.Tensor:
    """Penalize the rate of change of selected action terms."""
    start = 0
    action_slices = []
    for term_name, term_dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
        stop = start + term_dim
        if term_name in action_names:
            action_slices.append(slice(start, stop))
        start = stop

    if not action_slices:
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = torch.zeros(env.num_envs, device=env.device)
    for action_slice in action_slices:
        action_delta = env.action_manager.action[:, action_slice] - env.action_manager.prev_action[:, action_slice]
        action_rate += torch.sum(torch.square(action_delta), dim=1)
    return action_rate

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
    target_height: float, #초기 위치의 x,y를 유지하고 도달할 목표 z 높이
    slope: float = 2.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]

    object_pos_w = object.data.root_pos_w[:, 0:3]

    if hasattr(env, "_object_initial_pos") and object_cfg.name in env._object_initial_pos:
        initial_pos = env._object_initial_pos[object_cfg.name]
    elif hasattr(env, "_object_initial_xy") and object_cfg.name in env._object_initial_xy:
        initial_pos = object_pos_w.clone()
        initial_pos[:, 0:2] = env._object_initial_xy[object_cfg.name]
    else:
        initial_pos = object_pos_w

    target_pos = initial_pos.clone()
    target_pos[:, 2] = initial_pos[:, 2] + target_height
    distance_to_target = torch.norm(object_pos_w - target_pos, dim=1)

    normalized_distance = distance_to_target / target_height
    lift_progress = torch.clamp(1.0 - normalized_distance, min=0.0, max=1.0)
    initial_reward = math.exp(-slope)
    reward = (1.0 - torch.exp(-slope * lift_progress)) / (1.0 - initial_reward)
    return torch.clamp(reward, min=0.0, max=1.0)

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
