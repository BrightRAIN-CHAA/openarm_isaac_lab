# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv


class BimanualLiftIPPOEnv(ManagerBasedRLEnv):
    """Thin multi-agent adapter over the manager-based bimanual lift task.

    The underlying manager-based task still owns simulation, actions,
    observations, rewards and resets. This adapter only exposes a two-agent
    interface for SKRL IPPO/MAPPO: one agent controls the left arm/gripper and
    the other controls the right arm/gripper.
    """

    possible_agents = ["left_arm", "right_arm"]
    agents = possible_agents

    _AGENT_ACTION_TERMS = {
        "left_arm": ("left_arm_action", "left_gripper_action"),
        "right_arm": ("right_arm_action", "right_gripper_action"),
    }
    _AGENT_REWARD_TERMS = {
        "left_arm": (
            "left_reaching_object",
            "left_gripper_grasped_bonus",
            # "left_lifted_gate",
            "left_object_goal_tracking",
            "left_object_goal_tracking_fine_grained",
            "left_action_rate",
            "left_joint_vel",
        ),
        "right_arm": (
            "right_reaching_object",
            "right_gripper_grasped_bonus",
            # "right_lifted_gate",
            "right_object_goal_tracking",
            "right_object_goal_tracking_fine_grained",
            "right_action_rate",
            "right_joint_vel",
        ),
    }

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._agent_action_slices = self._build_agent_action_slices()
        policy_observation_space = self._policy_observation_space()
        self.action_spaces = {
            agent: gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self._agent_action_dim(agent),),
            )
            for agent in self.possible_agents
        }
        self.observation_spaces = {agent: policy_observation_space for agent in self.possible_agents}
        self.state_space = policy_observation_space

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    def reset(self, *args, **kwargs):
        observations, extras = super().reset(*args, **kwargs)
        return self._agent_observations(observations), extras

    def step(self, actions):
        if not isinstance(actions, Mapping):
            observations, reward, terminated, truncated, extras = super().step(actions)
            return observations, reward, terminated, truncated, extras

        observations, reward, terminated, truncated, extras = super().step(self._merge_agent_actions(actions))
        return (
            self._agent_observations(observations),
            self._agent_rewards(reward),
            self._agent_dones(terminated),
            self._agent_dones(truncated),
            extras,
        )

    def state(self):
        observations = self.observation_manager.compute()
        return self._policy_observation(observations)

    def _build_agent_action_slices(self) -> dict[str, list[slice]]:
        term_slices = {}
        start = 0
        for term_name, term_dim in zip(self.action_manager.active_terms, self.action_manager.action_term_dim):
            stop = start + term_dim
            term_slices[term_name] = slice(start, stop)
            start = stop

        agent_slices = {}
        for agent, term_names in self._AGENT_ACTION_TERMS.items():
            agent_slices[agent] = [term_slices[name] for name in term_names if name in term_slices]
        return agent_slices

    def _agent_action_dim(self, agent: str) -> int:
        return sum(action_slice.stop - action_slice.start for action_slice in self._agent_action_slices[agent])

    def _policy_observation_space(self):
        try:
            return self.single_observation_space["policy"]
        except (AttributeError, KeyError, TypeError):
            pass
        try:
            return self.observation_space["policy"]
        except (AttributeError, KeyError, TypeError):
            return self.observation_space

    def _merge_agent_actions(self, actions: Mapping[str, torch.Tensor]) -> torch.Tensor:
        merged = torch.zeros((self.num_envs, self.action_manager.total_action_dim), device=self.device)
        for agent, action in actions.items():
            cursor = 0
            action = action.to(self.device)
            for action_slice in self._agent_action_slices[agent]:
                width = action_slice.stop - action_slice.start
                merged[:, action_slice] = action[:, cursor : cursor + width]
                cursor += width
        return merged

    def _agent_observations(self, observations) -> dict[str, torch.Tensor]:
        policy_obs = self._policy_observation(observations)
        return {agent: policy_obs for agent in self.possible_agents}

    def _policy_observation(self, observations):
        if isinstance(observations, torch.Tensor):
            return observations
        if isinstance(observations, dict):
            if "policy" in observations:
                return self._policy_observation(observations["policy"])
            if "observations" in observations:
                return self._policy_observation(observations["observations"])
            if len(observations) == 1:
                return self._policy_observation(next(iter(observations.values())))
            return torch.cat([self._policy_observation(value).view(self.num_envs, -1) for value in observations.values()], dim=-1)
        return observations

    def _agent_rewards(self, fallback_reward: torch.Tensor) -> dict[str, torch.Tensor]:
        term_names = getattr(self.reward_manager, "_term_names", [])
        step_reward = getattr(self.reward_manager, "_step_reward", None)
        if step_reward is None:
            fallback_reward = fallback_reward.view(self.num_envs, -1)
            if fallback_reward.shape[1] != 1:
                fallback_reward = torch.sum(fallback_reward, dim=1, keepdim=True)
            return {agent: fallback_reward for agent in self.possible_agents}

        rewards = {}
        for agent, agent_terms in self._AGENT_REWARD_TERMS.items():
            agent_reward = torch.zeros(self.num_envs, device=self.device)
            found_term = False
            for term_name in agent_terms:
                if term_name not in term_names:
                    continue
                term_idx = term_names.index(term_name)
                agent_reward += step_reward[:, term_idx] * self.step_dt
                found_term = True
            rewards[agent] = agent_reward.view(self.num_envs, 1) if found_term else torch.zeros((self.num_envs, 1), device=self.device)
        return rewards

    def _agent_dones(self, done: torch.Tensor) -> dict[str, torch.Tensor]:
        done = done.view(self.num_envs, -1)
        if done.shape[1] != 1:
            done = torch.any(done, dim=1, keepdim=True)
        return {agent: done for agent in self.possible_agents}
