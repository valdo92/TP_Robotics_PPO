#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria
#
# Version modifiée : entraînement sans pybullet (utilise CartPole-v1 comme fallback)

import argparse
import datetime
import sys
import os
import random
import string
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Callable, List
import importlib.util

import gin
import gymnasium
import numpy as np
import stable_baselines3
from define_reward import DefineReward
from settings import EnvSettings, PPOSettings, TrainingSettings
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from torch import nn
from upkie.utils.spdlog import logging

TRAINING_PATH = os.environ.get("UPKIE_TRAINING_PATH", tempfile.gettempdir())


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default="",
        type=str,
        help="name of the new policy to train",
    )
    parser.add_argument(
        "--nb-envs",
        default=1,
        type=int,
        help="number of parallel simulation processes to run",
    )
    parser.add_argument(
        "--show",
        default=False,
        action="store_true",
        help="show simulator during trajectory rollouts (ignored for CartPole fallback)",
    )
    return parser.parse_args()


class InitRandomizationCallback(BaseCallback):
    def __init__(
        self,
        vec_env: VecEnv,
        key: str,
        max_value: float,
        start_timestep: int,
        end_timestep: int,
    ):
        super().__init__()
        self.end_timestep = end_timestep
        self.key = key
        self.max_value = max_value
        self.start_timestep = start_timestep
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        # schedule progress in [0,1]
        progress: float = np.clip(
            (self.num_timesteps - self.start_timestep) / self.end_timestep,
            0.0,
            1.0,
        )
        cur_value = progress * self.max_value
        # try to call update_init_rand if env implements it
        try:
            self.vec_env.env_method("update_init_rand", **{self.key: cur_value})
            self.logger.record(f"init_rand/{self.key}", cur_value)
        except Exception:
            # environment doesn't provide this hook: silently ignore
            pass
        return True


class RewardCallback(BaseCallback):
    def __init__(self, vec_env: VecEnv):
        super().__init__()
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        # Attempt to log env-specific reward components if they exist.
        for term in ("position_reward", "velocity_penalty", "action_change_penalty"):
            try:
                vals = self.vec_env.get_attr(f"last_{term}")
                reward = np.mean(vals) if len(vals) > 0 else 0.0
                self.logger.record(f"rewards/{term}", reward)
            except Exception:
                # attribute not present on the environment
                pass
        try:
            last_rewards = self.vec_env.get_attr("last_reward")
            reward = np.mean(last_rewards) if len(last_rewards) > 0 else 0.0
            self.logger.record("rewards/reward", reward)
        except Exception:
            # fallback: nothing to log
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
            (formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat)),
            None,
        )

    def _on_step(self) -> bool:
        # We wait for the first call to log operative config so that parameters
        # for functions called by the environment are logged as well.
        if self.n_calls != 1:
            return True

        # Use a dummy env to instantiate DefineReward for gin operative config
        with gymnasium.make("CartPole-v1") as dummy_env:
            _ = DefineReward(dummy_env)  # for the gin operative config
        print("Gin operative config:", gin.operative_config_str())
        if self.tb_formatter:
            self.tb_formatter.writer.add_text(
                "gin/operative_config",
                gin.operative_config_str(),
                global_step=None,
            )
        gin_path = f"{self.save_path}/operative_config.gin"
        with open(gin_path, "w") as fh:
            fh.write(gin.operative_config_str())
        logging.info(f"Saved gin configuration to {gin_path}")
        return True


def get_random_word():
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(8))


def init_env(max_episode_duration: float, show: bool, spine_path: str = ""):
    """
    Initialization factory WITHOUT pybullet/spine.
    Uses CartPole-v1 as a simple Gym fallback environment.
    If you have a pure-Python environment (Gym-compatible) replace 'CartPole-v1' by its id.
    """
    env_settings = EnvSettings()
    seed = random.randint(0, 1_000_000)

    def _init():
        # create a Gym environment; adjust max steps using agent_frequency if available
        try:
            agent_frequency = env_settings.agent_frequency
            max_steps = int(max_episode_duration * agent_frequency)
        except Exception:
            max_steps = None

        # Create CartPole with a max_episode_steps override when possible
        if max_steps is not None:
            # Gymnasium accepts max_episode_steps as an argument to make
            env = gymnasium.make("CartPole-v1", max_episode_steps=max_steps)
        else:
            env = gymnasium.make("CartPole-v1")

        env.reset(seed=seed)
        # Wrap with Monitor for SB3 logging
        return Monitor(env)

    set_random_seed(seed)
    return _init


def find_save_path(training_dir: str, policy_name: str):
    def path_for_iter(nb_iter: int):
        return Path(training_dir) / f"{policy_name}_{nb_iter}"

    nb_iter = 1
    while Path(path_for_iter(nb_iter)).exists():
        nb_iter += 1
    return path_for_iter(nb_iter)


def affine_schedule(y_0: float, y_1: float) -> Callable[[float], float]:
    diff = y_1 - y_0

    def schedule(x: float) -> float:
        return y_0 + x * diff

    return schedule


class DummyRunfiles:
    def Rlocation(self, path):
        if path.startswith("ppo_balancer/"):
            return path.replace("ppo_balancer/", "", 1)
        return path


def train_policy(policy_name: str, nb_envs: int, show: bool) -> None:
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    training_dir = Path(TRAINING_PATH) / date
    logging.info("Logging training data in %s", training_dir)
    logging.info("To track in TensorBoard, run " f"`tensorboard --logdir {training_dir}`")
    today_path = Path(TRAINING_PATH) / "today"
    target_path = Path(TRAINING_PATH) / date
    # ensure today symlink points to today's directory
    today_path.unlink(missing_ok=True)
    try:
        target_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    today_path.symlink_to(target_path)

    if policy_name == "":
        policy_name = get_random_word()
    logging.info('New policy name is "%s"', policy_name)

    training = TrainingSettings()
    deez_runfiles = DummyRunfiles()
    # spine_path not used in this no-pybullet variant; keep variable for compatibility
    spine_path = ""

    # Create vectorized environments (CartPole fallback)
    if nb_envs > 1:
        vec_env = SubprocVecEnv(
            [
                init_env(
                    max_episode_duration=training.max_episode_duration,
                    show=show,
                    spine_path=spine_path,
                )
                for _ in range(nb_envs)
            ],
            start_method="fork",
        )
    else:
        vec_env = DummyVecEnv(
            [
                init_env(
                    max_episode_duration=training.max_episode_duration,
                    show=show,
                    spine_path=spine_path,
                )
            ]
        )

    env_settings = EnvSettings()
    # dt/gamma calculation: if agent_frequency not present in fallback, use default
    try:
        dt = 1.0 / env_settings.agent_frequency
    except Exception:
        dt = 1.0 / 50.0
    gamma = 1.0 - dt / training.return_horizon
    logging.info(
        "Discount factor gamma=%f for a return horizon of %f s", gamma, training.return_horizon
    )

    ppo_settings = PPOSettings()
    policy = stable_baselines3.PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=affine_schedule(
            y_1=ppo_settings.learning_rate,  # progress_remaining=1.0
            y_0=ppo_settings.learning_rate / 3,  # progress_remaining=0.0
        ),
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
        tensorboard_log=str(training_dir),
        policy_kwargs={
            "activation_fn": nn.Tanh,
            "net_arch": {
                "pi": ppo_settings.net_arch_pi,
                "vf": ppo_settings.net_arch_vf,
            },
        },
        device="cpu",
        verbose=1,
    )

    save_path = find_save_path(training_dir, policy_name)
    logging.info("Training data will be logged to %s", save_path)

    try:
        policy.learn(
            total_timesteps=training.total_timesteps,
            callback=[
                CheckpointCallback(
                    save_freq=max(210_000 // max(nb_envs, 1), 1_000),
                    save_path=save_path,
                    name_prefix="checkpoint",
                ),
                SummaryWriterCallback(vec_env, str(save_path)),
                InitRandomizationCallback(
                    vec_env,
                    "pitch",
                    training.init_rand.get("pitch", 0.0),
                    start_timestep=0,
                    end_timestep=1e5,
                ),
                InitRandomizationCallback(
                    vec_env,
                    "v_x",
                    training.init_rand.get("v_x", 0.0),
                    start_timestep=0,
                    end_timestep=1e5,
                ),
                InitRandomizationCallback(
                    vec_env,
                    "omega_y",
                    training.init_rand.get("omega_y", 0.0),
                    start_timestep=0,
                    end_timestep=1e5,
                ),
                RewardCallback(vec_env),
            ],
            tb_log_name=policy_name,
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted.")

    # Save policy no matter what!
    os.makedirs(save_path, exist_ok=True)
    policy.save(f"{save_path}/final.zip")
    try:
        policy.env.close()
    except Exception:
        pass
    write_policy_makefile(str(save_path))
    deploy_policy(str(save_path))


def deploy_policy(policy_path: str):
    deployment_path = Path(TRAINING_PATH).parent / "policy"
    logging.info("Deploying policy from %s to %s", policy_path, deployment_path)
    os.makedirs(deployment_path, exist_ok=True)
    shutil.copy(f"{policy_path}/final.zip", f"{deployment_path}/params.zip")
    # operative_config.gin may not exist in some cases; copy if present
    gin_path_src = Path(policy_path) / "operative_config.gin"
    if gin_path_src.exists():
        shutil.copy(str(gin_path_src), f"{deployment_path}/operative_config.gin")


def write_policy_makefile(policy_path: str):
    makefile_path = f"{policy_path}/Makefile"
    logging.info("Saved policy Makefile to %s", makefile_path)
    with open(makefile_path, "w") as makefile:
        makefile.write(
            """# Makefile

help:
\t@echo "Usage: `make deploy` to deploy the policy"

deploy:
\tcp -f $(CURDIR)/final.zip ../../../data/params.zip
\tcp -f $(CURDIR)/operative_config.gin ../../../data/operative_config.gin"""
        )


if __name__ == "__main__":
    args = parse_command_line_arguments()
    agent_dir = Path(__file__).parent.parent
    # keep gin parsing if you have a config.gin; otherwise this will error
    try:
        gin.parse_config_file(str(agent_dir / "config.gin"))
    except Exception:
        logging.info("No config.gin parsed (missing or invalid); continuing with defaults.")
    train_policy(args.name, nb_envs=args.nb_envs, show=args.show)
