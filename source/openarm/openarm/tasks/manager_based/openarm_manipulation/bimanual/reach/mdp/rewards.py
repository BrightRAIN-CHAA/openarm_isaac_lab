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

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    distance = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    return 1.0 - torch.tanh(distance / std)

def gripper_is_grasped_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    action_name: str,
    contact_force_threshold: float = 20.0,
    near_distance_threshold: float = 0.025,
) -> torch.Tensor:
    sensor_names = (
        ("left_left_finger_contact", "left_right_finger_contact")
        if "left" in action_name
        else ("right_left_finger_contact", "right_right_finger_contact")
    )

    dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    near_reward = (dist < near_distance_threshold).float()
    contact_reward = (_max_contact_force(env, sensor_names) > contact_force_threshold).float()
    return contact_reward * near_reward

def object_lifted_gate(
    env: ManagerBasedRLEnv,
    minimal_lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return 1 when the object is lifted above reset height."""
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, 0:3]
    initial_pos = _object_initial_pos(env, object_pos_w, object_cfg)
    return (object_pos_w[:, 2] >= initial_pos[:, 2] + minimal_lift_height).float()

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    minimal_lift_height: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward object tracking to the commanded goal position after minimum lift height."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos_w = object.data.root_pos_w[:, 0:3]
    initial_pos = _object_initial_pos(env, object_pos_w, object_cfg)
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])

    distance = torch.norm(des_pos_w - object_pos_w, dim=1)
    goal_reward = 1.0 - torch.tanh(distance / std)

    height_gate = object_pos_w[:, 2] >= initial_pos[:, 2] + minimal_lift_height

    return torch.clamp(height_gate.float() * goal_reward, min=0.0)

def action_terms_rate_l2(
    env: ManagerBasedRLEnv,
    action_names: tuple[str, ...],
) -> torch.Tensor:
    """Penalize the rate of change for selected action terms."""
    start = 0
    action_rate = torch.zeros(env.num_envs, device=env.device)

    for term_name, term_dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
        stop = start + term_dim
        if term_name in action_names:
            action_delta = env.action_manager.action[:, start:stop] - env.action_manager.prev_action[:, start:stop]
            action_rate += torch.sum(torch.square(action_delta), dim=1)
        start = stop

    return action_rate

# Helper functions for reward calculations

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


def _max_contact_force(env: ManagerBasedRLEnv, sensor_names: tuple[str, ...]) -> torch.Tensor:
    contact_forces = []
    for sensor_name in sensor_names:
        contact_force_mag = _contact_force_magnitudes(env, sensor_name)
        if contact_force_mag.shape[1] > 0:
            contact_forces.append(torch.max(contact_force_mag, dim=1).values)

    if not contact_forces:
        return torch.zeros(env.num_envs, device=env.device)

    return torch.max(torch.stack(contact_forces, dim=1), dim=1).values


def _ee_object_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = object.data.root_pos_w[:, 0:3]
    frame_idx = ee_frame_cfg.body_ids[0]
    ee_pos_w = ee_frame.data.target_pos_w[:, frame_idx, :]
    return torch.norm(object_pos_w - ee_pos_w, dim=1)

def _object_initial_pos(
    env: ManagerBasedRLEnv,
    object_pos_w: torch.Tensor,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    if hasattr(env, "_object_initial_pos") and object_cfg.name in env._object_initial_pos:
        return env._object_initial_pos[object_cfg.name]

    if hasattr(env, "_object_initial_xy") and object_cfg.name in env._object_initial_xy:
        initial_pos = object_pos_w.clone()
        initial_pos[:, 0:2] = env._object_initial_xy[object_cfg.name]
        return initial_pos

    return object_pos_w
