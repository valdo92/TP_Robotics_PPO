#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

import argparse
import datetime
import pytz
import uuid
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
# Add this with your other imports
from stable_baselines3.common.vec_env import VecNormalize

# --- IMPORT YOUR WRAPPER ---
# Make sure your file is named 'wrapper.py' and in the 'ppo_balancer' folder
from ppo_balancer.wrapper import NewWrapper 

upkie.envs.register()

TRAINING_PATH = os.environ.get("UPKIE_TRAINING_PATH", tempfile.gettempdir())

import gymnasium as gym
import numpy as np

class DefineReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.sigma_pitch = 0.15
        self.weight_velocity = 0.1 
        self.weight_action = 0.1
        
        # Initialize memory for "Action Change Penalty"
        self.last_action = None
        
        # Initialize logs so train.py doesn't crash
        self.last_position_reward = 0.0
        self.last_velocity_penalty = 0.0
        self.last_action_change_penalty = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Extract from NewWrapper Observation (12 dims)
        # Index 0 = Pitch
        # Index 3 = Ground Velocity
        pitch = obs[0]
        ground_vel = obs[3]

        # 2. Calculate Components
        # Position: Bell curve around 0
        pos_reward = np.exp(- (pitch**2) / (2 * self.sigma_pitch**2))
        
        # Velocity: Penalize moving fast
        vel_penalty = -1.0 * abs(ground_vel)
        
        # Action: Penalize twitching
        if self.last_action is not None:
            act_penalty = -1.0 * np.sum(np.abs(action - self.last_action))
        else:
            act_penalty = 0.0
        self.last_action = action.copy()

        # 3. Save for TensorBoard (train.py reads these!)
        self.last_position_reward = pos_reward
        self.last_velocity_penalty = vel_penalty
        self.last_action_change_penalty = act_penalty

        # 4. Add to existing reward (Alive Bonus from wrapper.py)
        # We multiply by weights here
        total_reward = (
            reward + 
            pos_reward + 
            (self.weight_velocity * vel_penalty) + 
            (self.weight_action * act_penalty)
        )

        return obs, total_reward, terminated, truncated, info

def cleanup(vec_env):
    try:
        vec_env.close()
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

# =============================================================================
# FORCE LEARNING WRAPPER (Unchanged from Q3)
# =============================================================================
class ForceLearningWrapper(gymnasium.Wrapper):
    """
    Adds external forces to the robot base during training.
    """
    def __init__(self, env, max_force=0.0, push_probability=0.02, push_duration=1.0):
        super().__init__(env)
        self.max_force = max_force
        self.push_probability = push_probability
        self.push_duration = push_duration 
        self._current_force = np.zeros(3)
        self._push_timer = 0.0
        self.dt = env.unwrapped.dt if hasattr(env.unwrapped, "dt") else 0.005

    def set_max_force(self, max_force: float):
        self.max_force = max_force

    def step(self, action):
        if self._push_timer > 0:
            self._push_timer -= self.dt
            if self._push_timer <= 0:
                self._current_force = np.zeros(3)
                self.unwrapped.set_bullet_action({}) 
        
        elif self.max_force > 0.1 and random.random() < self.push_probability:
            self._push_timer = self.push_duration
            # Randomize direction (Sagittal push: +X or -X)
            force_mag = random.uniform(self.max_force * 0.5, self.max_force)
            direction = 1 if random.random() > 0.5 else -1
            self._current_force = np.array([force_mag * direction, 0.0, 0.0])

        if self._push_timer > 0:
            # This works with UpkieServos too!
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
# CURRICULUM CALLBACK (Unchanged from Q3)
# =============================================================================
class ForceCurriculumCallback(BaseCallback):
    def __init__(self, vec_env: VecEnv, target_max_force: float, start_timestep: int, end_timestep: int):
        super().__init__()
        self.vec_env = vec_env
        self.target_max_force = target_max_force
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep

    def _on_step(self) -> bool:
        if self.num_timesteps < self.start_timestep:
            progress = 0.0
        else:
            progress = np.clip((self.num_timesteps - self.start_timestep) / (self.end_timestep - self.start_timestep), 0.0, 1.0)
        
        current_max = progress * self.target_max_force
        self.vec_env.env_method("set_max_force", max_force=current_max)
        self.logger.record("curriculum/max_force", current_max)
        return True

# =============================================================================
# HELPERS
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
        if self.tb_formatter:
            self.tb_formatter.writer.add_text("gin/operative_config", gin.operative_config_str(), global_step=None)
        gin_path = f"{self.save_path}/operative_config.gin"
        with open(gin_path, "w") as fh:
            fh.write(gin.operative_config_str())
        logging.info(f"Saved gin configuration to {gin_path}")
        return True

def get_random_word():
    paris_tz = pytz.timezone("Europe/Paris")
    now = datetime.datetime.now(paris_tz)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{unique_suffix}"

def get_bullet_argv(shm_name: str, show: bool) -> List[str]:
    env_settings = EnvSettings()
    agent_frequency = env_settings.agent_frequency
    spine_frequency = env_settings.spine_frequency
    nb_substeps = spine_frequency / agent_frequency
    bullet_argv = ["--shm-name", shm_name, "--nb-substeps", str(nb_substeps), "--spine-frequency", str(spine_frequency)]
    if show: bullet_argv.append("--show")
    return bullet_argv

# =============================================================================
# ENVIRONMENT INIT (MODIFIED FOR Q4 + FORCE)
# =============================================================================

def init_env(max_episode_duration: float, show: bool, spine_path: str):
    env_settings = EnvSettings()
    seed = random.randint(0, 1_000_000)

    def _init():
        # 1. Start the Spine Process
        shm_name = f"/{get_random_word()}"
        pid = os.fork()
        if pid == 0:  # child process: spine
            argv = get_bullet_argv(shm_name, show=show)
            os.execvp(spine_path, ["bullet"] + argv)
            return

        # 2. Setup the Parent Process (Trainer)
        agent_frequency = env_settings.agent_frequency
        
        # A. Base Environment: UpkieServos (Q4)
        base_env = upkie.envs.UpkieServos(
            frequency=agent_frequency,
            shm_name=shm_name,  # Connect to the forked spine
            # We don't need other params here as UpkieServos defaults are fine for raw control
        )
        base_env.reset(seed=seed)
        
        # Cleanup magic for forked process
        base_env._prepatch_close = base_env.close
        def close_monkeypatch():
            try:
                os.kill(pid, signal.SIGINT)
                os.waitpid(pid, 0)
            except:
                pass
            finally:
                base_env._prepatch_close()
        base_env.close = close_monkeypatch
        
        # B. Apply Your New Wrapper (Q4)
        # (This handles the vector observation and 5-dim action space)
        env = NewWrapper(base_env)
        
        # C. Apply Reward (Q4)
        # (Needed because we aren't using wrap_velocity_env anymore)
        env = DefineReward(env)
        
        # D. Apply Force Learning (Q3)
        # (Adds the pushes)
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
    if policy_name == "": policy_name = get_random_word()
    
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
    # Phase 1: Learn to balance (0 to 200k)
    # Phase 2: Learn to resist forces (200k to 800k)
    TOTAL_STEPS = training.total_timesteps
    FORCE_START = 400_000
    FORCE_END = 800_000
    MAX_TRAIN_FORCE = 12.0

    try:
        policy.learn(
            total_timesteps=TOTAL_STEPS,
            callback=[
                CheckpointCallback(save_freq=max(210_000 // nb_envs, 1_000), save_path=save_path, name_prefix="checkpoint"),
                SummaryWriterCallback(vec_env, save_path),
                InitRandomizationCallback(vec_env, "pitch", training.init_rand["pitch"], 2e5, 1e5),
                InitRandomizationCallback(vec_env, "v_x", training.init_rand["v_x"], 2e5, 1e5),
                InitRandomizationCallback(vec_env, "omega_y", training.init_rand["omega_y"], 2e5, 1e5),
                
                # Force Curriculum
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
    try:
        shutil.copy(f"{policy_path}/final.zip", f"{TRAINING_PATH}/../policy/params.zip")
        shutil.copy(f"{policy_path}/operative_config.gin", f"{TRAINING_PATH}/../policy/operative_config.gin")
    except:
        pass

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