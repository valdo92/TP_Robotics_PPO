#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

import argparse
import datetime
import os
import random
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Callable, List
import atexit

import gin
import gymnasium
import numpy as np
import stable_baselines3
import upkie.envs
import upkie.envs.rewards
from define_reward import DefineReward
from rules_python.python.runfiles import runfiles
from settings import EnvSettings, PPOSettings, TrainingSettings
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from torch import nn
from upkie.utils.spdlog import logging
from wrap_velocity_env import wrap_velocity_env

upkie.envs.register()

def cleanup(vec_env):
    try:
        vec_env.close()
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

TRAINING_PATH = os.environ.get("UPKIE_TRAINING_PATH", tempfile.gettempdir())


# =============================================================================
# NEW: Force Learning Wrapper
# =============================================================================
class ForceLearningWrapper(gymnasium.Wrapper):
    def __init__(self, env, max_force=0.0, push_probability=0.02, push_duration=1.0):
        super().__init__(env)
        self.max_force = max_force
        self.push_probability = push_probability
        self.push_duration = push_duration
        self._current_force = np.zeros(3)
        self._push_timer = 0.0
        self.dt = env.unwrapped.dt if hasattr(env.unwrapped, "dt") else 0.005

        # NEW: Delay after reset before applying forces
        self.reset_delay = 0.5  # seconds
        self._time_since_reset = 0.0
    def set_max_force(self, max_force: float):
        """Called by the Callback to update difficulty."""
        self.max_force = max_force

    def reset(self, **kwargs):
        self._time_since_reset = 0.0
        self._push_timer = 0.0
        self._current_force = np.zeros(3)
        self.unwrapped.set_bullet_action({})
        return super().reset(**kwargs)

    def step(self, action):

        # NEW: accumulate time after reset
        self._time_since_reset += self.dt

        # NEW: no pushes during the delay period
        if self._time_since_reset < self.reset_delay:
            return super().step(action)

        # (existing logic)
        if self._push_timer > 0:
            self._push_timer -= self.dt
            if self._push_timer <= 0:
                self._current_force = np.zeros(3)
                self.unwrapped.set_bullet_action({})
        elif self.max_force > 0.1 and random.random() < self.push_probability:
            self._push_timer = self.push_duration
            force_mag = random.uniform(self.max_force * 0.5, self.max_force)
            direction = 1 if random.random() > 0.5 else -1
            self._current_force = np.array([force_mag * direction, 0.0, 0.0])

        if self._push_timer > 0:
            self.unwrapped.set_bullet_action({
                "external_forces": {
                    "base": {
                        "force": self._current_force.tolist(),
                        "position": [0.0, 0.0, 0.0],
                    }
                }
            })

        return super().step(action)



# =============================================================================
# NEW: Force Curriculum Callback
# =============================================================================
class ForceCurriculumCallback(BaseCallback):
    """
    Linearly increases the max_force applied to the robot over time.
    """
    def __init__(
        self,
        vec_env: VecEnv,
        target_max_force: float,
        start_timestep: int,
        end_timestep: int,
    ):
        super().__init__()
        self.vec_env = vec_env
        self.target_max_force = target_max_force
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep

    def _on_step(self) -> bool:
        # 1. Calculate progress (0.0 to 1.0)
        if self.num_timesteps < self.start_timestep:
            progress = 0.0
        else:
            progress = np.clip(
                (self.num_timesteps - self.start_timestep) / (self.end_timestep - self.start_timestep),
                0.0,
                1.0,
            )
        
        # 2. Calculate current difficulty
        current_max = progress * self.target_max_force
        
        # 3. Update the environment
        # We use env_method to reach into the ForceLearningWrapper inside the SubprocVecEnv
        self.vec_env.env_method("set_max_force", max_force=current_max)
        
        # 4. Log
        self.logger.record("curriculum/max_force", current_max)
        return True


# =============================================================================
# EXISTING HELPERS (Unchanged)
# =============================================================================

def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="", type=str, help="name of the new policy to train")
    parser.add_argument("--nb-envs", default=1, type=int, help="number of parallel simulation processes")
    parser.add_argument("--show", default=False, action="store_true", help="show simulator")
    return parser.parse_args()

class InitRandomizationCallback(BaseCallback):
    def __init__(self, vec_env: VecEnv, key: str, max_value: float, start_timestep: int, end_timestep: int):
        super().__init__()
        self.end_timestep = end_timestep
        self.key = key
        self.max_value = max_value
        self.start_timestep = start_timestep
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        progress: float = np.clip((self.num_timesteps - self.start_timestep) / self.end_timestep, 0.0, 1.0)
        cur_value = progress * self.max_value
        self.vec_env.env_method("update_init_rand", **{self.key: cur_value})
        self.logger.record(f"init_rand/{self.key}", cur_value)
        return True

class RewardCallback(BaseCallback):
    def __init__(self, vec_env: VecEnv):
        super().__init__()
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        for term in ("position_reward", "velocity_penalty", "action_change_penalty"):
            if hasattr(self.vec_env, "get_attr"):
                 # Wrap in try/except or check if attr exists to be safe with SubProc
                 try:
                     vals = self.vec_env.get_attr(f"last_{term}")
                     self.logger.record(f"rewards/{term}", np.mean(vals))
                 except:
                     pass
        return True

class SummaryWriterCallback(BaseCallback):
    def __init__(self, vec_env: VecEnv, save_path: str):
        super().__init__()
        self.save_path = save_path
        self.vec_env = vec_env

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

    def _on_step(self) -> bool:
        if self.n_calls != 1: return True
        with gymnasium.make("CartPole-v1") as dummy_env:
            _ = DefineReward(dummy_env)
        print("Gin operative config:", gin.operative_config_str())
        self.tb_formatter.writer.add_text("gin/operative_config", gin.operative_config_str(), global_step=None)
        gin_path = f"{self.save_path}/operative_config.gin"
        with open(gin_path, "w") as fh:
            fh.write(gin.operative_config_str())
        logging.info(f"Saved gin configuration to {gin_path}")
        return True

def get_date_string():
    """Return a string like '2025-12-09_1530' for naming purposes."""
    now = datetime.datetime.now()
    return "Traning-"+ now.strftime("%Y-%m-%d_%H%M")

def get_bullet_argv(shm_name: str, show: bool) -> List[str]:
    env_settings = EnvSettings()
    agent_frequency = env_settings.agent_frequency
    spine_frequency = env_settings.spine_frequency
    nb_substeps = spine_frequency / agent_frequency
    bullet_argv = ["--shm-name", shm_name, "--nb-substeps", str(nb_substeps), "--spine-frequency", str(spine_frequency)]
    if show: bullet_argv.append("--show")
    return bullet_argv

# =============================================================================
# ENVIRONMENT INIT
# =============================================================================

def init_env(max_episode_duration: float, show: bool, spine_path: str):
    env_settings = EnvSettings()
    seed = random.randint(0, 1_000_000)

    def _init():
        shm_name = f"/{get_date_string()}"
        pid = os.fork()
        if pid == 0:  # child process: spine
            argv = get_bullet_argv(shm_name, show=show)
            os.execvp(spine_path, ["bullet"] + argv)
            return

        # parent process: trainer
        agent_frequency = env_settings.agent_frequency
        velocity_env = gymnasium.make(
            env_settings.env_id,
            max_episode_steps=int(max_episode_duration * agent_frequency),
            frequency=agent_frequency,
            regulate_frequency=False,
            shm_name=shm_name,
            spine_config=env_settings.spine_config,
            max_ground_velocity=env_settings.max_ground_velocity,
        )
        velocity_env.reset(seed=seed)
        velocity_env._prepatch_close = velocity_env.close

        def close_monkeypatch():
            try:
                os.kill(pid, signal.SIGINT)
                os.waitpid(pid, 0)
            except:
                pass
            finally:
                velocity_env._prepatch_close()

        velocity_env.close = close_monkeypatch
        
        # 1. Standard Wrapper
        env = wrap_velocity_env(velocity_env, env_settings, training=True)
        
        # 2. NEW: Force Learning Wrapper
        # We wrap the "wrapped" env again to add physics noise
        env = ForceLearningWrapper(env, max_force=0.0) 
        
        return Monitor(env)

    set_random_seed(seed)
    return _init


def find_save_path(training_dir: str, policy_name: str):
    nb_iter = 1
    while Path(training_dir / f"{policy_name}_{nb_iter}").exists():
        nb_iter += 1
    return Path(training_dir) / f"{policy_name}_{nb_iter}"


def affine_schedule(y_0: float, y_1: float) -> Callable[[float], float]:
    diff = y_1 - y_0
    def schedule(x: float) -> float:
        return y_0 + x * diff
    return schedule


def train_policy(policy_name: str, nb_envs: int, show: bool) -> None:
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    training_dir = Path(TRAINING_PATH) / date
    if policy_name == "": policy_name = get_date_string()
    
    training = TrainingSettings()
    deez_runfiles = runfiles.Create()
    spine_path = Path(agent_dir) / deez_runfiles.Rlocation("upkie/spines/bullet_spine")

    # Create Envs
    vec_env = SubprocVecEnv(
        [init_env(training.max_episode_duration, show, spine_path) for _ in range(nb_envs)],
        start_method="fork",
    ) if nb_envs > 1 else DummyVecEnv([init_env(training.max_episode_duration, show, spine_path)])

    atexit.register(cleanup, vec_env)

    env_settings = EnvSettings()
    dt = 1.0 / env_settings.agent_frequency
    gamma = 1.0 - dt / training.return_horizon

    ppo_settings = PPOSettings()
    policy = stable_baselines3.PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=affine_schedule(y_1=ppo_settings.learning_rate, y_0=ppo_settings.learning_rate / 3),
        n_steps=ppo_settings.n_steps,
        batch_size=ppo_settings.batch_size,
        n_epochs=ppo_settings.n_epochs,
        gamma=gamma,
        gae_lambda=ppo_settings.gae_lambda,
        clip_range=ppo_settings.clip_range,
        clip_range_vf=ppo_settings.clip_range_vf,
        normalize_advantage=ppo_settings.normalize_advantage,
        ent_coef=ppo_settings.ent_coef,
        vf_coef=ppo_settings.vf_coef,
        max_grad_norm=ppo_settings.max_grad_norm,
        use_sde=ppo_settings.use_sde,
        sde_sample_freq=ppo_settings.sde_sample_freq,
        target_kl=ppo_settings.target_kl,
        tensorboard_log=training_dir,
        policy_kwargs={
            "activation_fn": nn.Tanh,
            "net_arch": {"pi": ppo_settings.net_arch_pi, "vf": ppo_settings.net_arch_vf},
        },
        device="cpu",
        verbose=1,
    )

    save_path = find_save_path(training_dir, policy_name)

    # --- DEFINE CURRICULUM TIMELINE ---
    # Phase 1: Learn to balance (0 to 100k)
    # Phase 2: Learn to resist forces (100k to 400k)
    TOTAL_STEPS = training.total_timesteps
    FORCE_START = 400_000
    FORCE_END = 800_000 # Ramps up force from 100k to 400k steps
    MAX_TRAIN_FORCE = 5.0 # Newtons (A bit lower than max test force to ensure stability)

    try:
        policy.learn(
            total_timesteps=TOTAL_STEPS,
            callback=[
                CheckpointCallback(save_freq=max(210_000 // nb_envs, 1_000), save_path=save_path, name_prefix="checkpoint"),
                SummaryWriterCallback(vec_env, save_path),
                
                # 1. Existing Initialization Randomization
                InitRandomizationCallback(vec_env, "pitch", training.init_rand["pitch"], 0, 1e5),
                InitRandomizationCallback(vec_env, "v_x", training.init_rand["v_x"], 0, 1e5),
                InitRandomizationCallback(vec_env, "omega_y", training.init_rand["omega_y"], 0, 1e5),
                
                # 2. NEW Force Curriculum
                ForceCurriculumCallback(
                    vec_env, 
                    target_max_force=MAX_TRAIN_FORCE, 
                    start_timestep=FORCE_START, 
                    end_timestep=FORCE_END
                ),
                
                RewardCallback(vec_env),
            ],
            tb_log_name=policy_name,
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted...")
    finally:
        os.makedirs(save_path, exist_ok=True)
        policy.save(f"{save_path}/final.zip")
        write_policy_makefile(save_path)
        deploy_policy(save_path)

def deploy_policy(policy_path: str):
    deployment_path = Path(TRAINING_PATH).parent / "policy"
    shutil.copy(f"{policy_path}/final.zip", f"{TRAINING_PATH}/../policy/params.zip")
    shutil.copy(f"{policy_path}/operative_config.gin", f"{TRAINING_PATH}/../policy/operative_config.gin")

def write_policy_makefile(policy_path: str):
    makefile_path = f"{policy_path}/Makefile"
    with open(makefile_path, "w") as makefile:
        makefile.write("""# Makefile
help:
\t@echo "Usage: `make deploy` to deploy the policy"
deploy:
\tcp -f $(CURDIR)/final.zip ../../../data/params.zip
\tcp -f $(CURDIR)/operative_config.gin ../../../data/operative_config.gin""")

if __name__ == "__main__":
    args = parse_command_line_arguments()
    agent_dir = Path(__file__).parent.parent
    gin.parse_config_file(str(agent_dir / "config.gin"))
    train_policy(args.name, nb_envs=args.nb_envs, show=args.show)