from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class exponential_modify_reward_weight(ManagerTermBase):
    """Curriculum that exponentially changes a reward weight over a fixed number of steps."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)
        self._initial_weight = self._term_cfg.weight

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        num_steps: int,
        end_rate_ratio: float = 4.0,
    ) -> float:
        progress = min(env.common_step_counter / num_steps, 1.0)

        if end_rate_ratio == 1.0:
            shaped_progress = progress
        else:
            exponent = math.log(end_rate_ratio)
            shaped_progress = (math.exp(exponent * progress) - 1.0) / (end_rate_ratio - 1.0)

        self._term_cfg.weight = self._initial_weight + (weight - self._initial_weight) * shaped_progress
        env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return self._term_cfg.weight


class linear_modify_reward_weight(ManagerTermBase):
    """Curriculum that linearly changes a reward weight over a fixed number of steps."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)
        self._initial_weight = self._term_cfg.weight

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        num_steps: int,
    ) -> float:
        progress = min(env.common_step_counter / num_steps, 1.0)
        self._term_cfg.weight = self._initial_weight + (weight - self._initial_weight) * progress
        env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return self._term_cfg.weight
