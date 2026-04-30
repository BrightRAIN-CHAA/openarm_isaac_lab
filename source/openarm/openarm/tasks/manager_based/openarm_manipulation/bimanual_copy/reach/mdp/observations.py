# Copyright 2025 Enactic, Inc.
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp # Isaac Lab 기본 MDP 함수들을 사용하기 위해 임포트
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """로봇 베이스 기준 물체의 위치를 계산합니다."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_quat_w = object.data.root_quat_w[:, :4]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w, object_quat_w
    )
    return object_pos_b

def get_bimanual_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """양팔의 관측치를 [num_envs, obs_dim * 2] 형태로 반환합니다."""
    all_obs = []
    
    # 1. 액션들 가져오기 (2048 규격)
    arm_actions = mdp.last_action(env, action_name="arm_action")
    gripper_actions = mdp.last_action(env, action_name="gripper_action")

    for side in ["left", "right"]:
        # 관절 정보 (각 22차원)
        joint_pos = mdp.joint_pos_rel(env, SceneEntityCfg("robot", joint_names=[f"openarm_{side}_joint[1-7]", f"openarm_{side}_finger.*"]))
        joint_vel = mdp.joint_vel_rel(env, SceneEntityCfg("robot", joint_names=[f"openarm_{side}_joint[1-7]", f"openarm_{side}_finger.*"]))

        # 물체 및 목표 위치 (각 3차원)
        obj_pos_b = object_position_in_robot_root_frame(env, SceneEntityCfg(f"object_{side}"))
        full_target_pos = mdp.generated_commands(env, command_name="object_pose")
        target_pos_b = full_target_pos[:, :3] if side == "left" else full_target_pos[:, 3:6]

        # 액션 슬라이싱 (양팔 분할)
        if side == "left":
            l_arm = arm_actions[:, :7]
            l_grip = gripper_actions[:, :gripper_actions.shape[1]//2]
            last_action = torch.cat([l_arm, l_grip], dim=-1)
        else:
            r_arm = arm_actions[:, 7:14]
            r_grip = gripper_actions[:, gripper_actions.shape[1]//2:]
            last_action = torch.cat([r_arm, r_grip], dim=-1)

        # 개별 팔 데이터 합치기 (64차원 내외)
        arm_data = torch.cat([joint_pos, joint_vel, obj_pos_b, target_pos_b, last_action], dim=-1)
        all_obs.append(arm_data)

    # [2048, 64] + [2048, 64] -> [2048, 128] 옆으로 길게 붙이기!
    return torch.cat(all_obs, dim=-1)