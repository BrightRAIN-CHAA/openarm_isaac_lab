import gymnasium as gym
from source.openarm.openarm.tasks.manager_based.openarm_manipulation.bimanual_copy.reach.config import agents

# 1. 학습용 환경 등록
gym.register(
    id="Lift_copy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # __package__를 써서 현재 config 폴더 내의 파일을 정확히 가리킵니다.
        "env_cfg_entry_point": f"{__package__}.joint_pos_env_cfg:OpenArmCubeLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLiftPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# 2. 플레이용 환경 등록
gym.register(
    id="Lift_copy_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__package__}.joint_pos_env_cfg:OpenArmCubeLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLiftPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)