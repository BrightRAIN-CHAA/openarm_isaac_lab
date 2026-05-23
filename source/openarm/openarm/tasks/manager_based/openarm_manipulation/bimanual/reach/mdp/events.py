from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def store_object_initial_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset 직후 object의 초기 3D 위치를 저장한다."""

    object: RigidObject = env.scene[object_cfg.name]

    if not hasattr(env, "_object_initial_pos"):
        env._object_initial_pos = {}

    if object_cfg.name not in env._object_initial_pos:
        env._object_initial_pos[object_cfg.name] = torch.zeros(
            env.num_envs,
            3,
            device=env.device,
        )

    env._object_initial_pos[object_cfg.name][env_ids] = object.data.root_pos_w[env_ids, 0:3].clone()


def store_object_initial_xy(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Backward-compatible alias for configs that still call the old event name."""

    store_object_initial_position(env, env_ids, object_cfg)
