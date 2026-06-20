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

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode (if supported by the agent class)
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    
    # --- [HYBRID STATE MEMORY] ---
    device = env.unwrapped.device if hasattr(env.unwrapped, "device") else "cuda:0"
    num_envs = env.unwrapped.num_envs if hasattr(env.unwrapped, "num_envs") else 1
    has_dropped_left = torch.zeros(num_envs, dtype=torch.bool, device=device)
    frozen_action_left = torch.zeros((num_envs, 7), device=device)
    
    has_dropped_right = torch.zeros(num_envs, dtype=torch.bool, device=device)
    frozen_action_right = torch.zeros((num_envs, 7), device=device)
    # -----------------------------
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            if hasattr(env, "possible_agents"):
                # - multi-agent (deterministic) actions
                outputs = runner.agent.act(observations=obs, states=obs, timestep=0, timesteps=0)
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                
                # --- [CUSTOM HYBRID CONTROL LOGIC] ---
                try:
                    isaac_env = env.unwrapped
                    if hasattr(isaac_env, "unwrapped"):
                        isaac_env = isaac_env.unwrapped
                        
                    if hasattr(isaac_env, "reset_buf"):
                        reset_envs = isaac_env.reset_buf
                        has_dropped_left[reset_envs] = False
                        has_dropped_right[reset_envs] = False
                    
                    left_obj_pos = isaac_env.scene["object_left"].data.root_pos_w
                    right_obj_pos = isaac_env.scene["object_right"].data.root_pos_w
                    
                    left_target_pose = isaac_env.command_manager.get_command("left_object_pose")
                    right_target_pose = isaac_env.command_manager.get_command("right_object_pose")
                    left_target_pos = left_target_pose[:, :3]
                    right_target_pos = right_target_pose[:, :3]
                    
                    left_dist = torch.norm(left_obj_pos - left_target_pos, dim=-1)
                    right_dist = torch.norm(right_obj_pos - right_target_pos, dim=-1)
                    
                    distance_threshold = 0.01                    
                    def get_freeze_action(env_obj, term_name):
                        term = env_obj.action_manager._terms[term_name]
                        robot = env_obj.scene["robot"]
                        
                        # 관절 인덱스 가져오기 (이전 코드에서 에러 안 났던 부분)
                        j_ids = term._joint_ids if hasattr(term, "_joint_ids") else term.joint_ids
                        
                        # IsaacLab 로봇 객체에서 기본 각도를 가져옵니다. (에러 발생했던 부분 완벽 수정!)
                        default_pos = robot.data.default_joint_pos[:, j_ids]
                        
                        # 스케일은 config에 정의된 0.5로 고정
                        scale = 0.5
                        
                        current_pos = robot.data.joint_pos[:, j_ids]
                        return (current_pos - default_pos) / scale
                    
                    # --- 왼쪽 팔 로직 ---
                    just_arrived_left = (left_dist < distance_threshold) & ~has_dropped_left
                    if just_arrived_left.any():
                        has_dropped_left[just_arrived_left] = True
                        # action_manager에서 정확한 관절 인덱스와 기본값을 역산해서 완벽한 제동 액션을 구합니다!
                        frozen_action_left[just_arrived_left] = get_freeze_action(isaac_env, "left_arm_action")[just_arrived_left]
                        
                    if has_dropped_left.any():
                        actions["left_arm"][has_dropped_left, :7] = frozen_action_left[has_dropped_left]
                        actions["left_arm"][has_dropped_left, 7] = 1.0  
                        
                    # --- 오른쪽 팔 로직 ---
                    just_arrived_right = (right_dist < distance_threshold) & ~has_dropped_right
                    if just_arrived_right.any():
                        has_dropped_right[just_arrived_right] = True
                        frozen_action_right[just_arrived_right] = get_freeze_action(isaac_env, "right_arm_action")[just_arrived_right]
                        
                    if has_dropped_right.any():
                        actions["right_arm"][has_dropped_right, :7] = frozen_action_right[has_dropped_right]
                        actions["right_arm"][has_dropped_right, 7] = 1.0  
                        
                except Exception as e:
                    print(f"하이브리드 로직 에러: {e}")
                    pass
                # -------------------------------------
            else:
                # - single-agent (deterministic) actions
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()