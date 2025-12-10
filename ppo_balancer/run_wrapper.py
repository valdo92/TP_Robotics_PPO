#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
from typing import Tuple

import gin
import gymnasium as gym
import numpy as np
import upkie.envs
from settings import EnvSettings, PPOSettings, TrainingSettings
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # <--- ADDED
from upkie.utils.raspi import configure_agent_process, on_raspi
from upkie.utils.robot_state import RobotState
from upkie.utils.robot_state_randomization import RobotStateRandomization

# --- IMPORT YOUR WRAPPER ---
from ppo_balancer.wrapper import NewWrapper 
from upkie.envs import UpkieServos # Import the base env class directly

upkie.envs.register()

def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        default=None,
        help="Path to the policy parameters file.",
    )
    parser.add_argument(
        "--training",
        default=False,
        action="store_true",
        help="add noise and actuation lag, as in training",
    )
    return parser.parse_args()

def run_episode(env: gym.Wrapper, policy, force_magnitude: float) -> bool:
    """
    Run one episode with a specific lateral force.
    Returns: True if successful (did not fall), False otherwise.
    """
    # Reset returns (obs, info)
    observation = env.reset() # In VecEnv, this returns just obs
    
    # Simulation parameters
    # Note: env is a VecEnv now, so we access the first env
    dt = env.envs[0].unwrapped.dt 
    simtime = 0.0
    
    # Force parameters
    FORCE_DURATION = 1.0  # Duration of the push
    RAMP_DURATION = 0.2   # Time to reach full force
    TIME_BEFORE_FORCE = 1.0
    TIME_AFTER_FORCE = 5.0
    force_active = True
    success = True
    
    # History for logging
    pitches = []

    while True:
        # 1. Predict Action using the PPO Policy
        action, _ = policy.predict(observation, deterministic=True)
        
        is_in_force_window = (
            simtime > TIME_BEFORE_FORCE and 
            simtime < TIME_BEFORE_FORCE + FORCE_DURATION
        )
        
        # 2. Handle External Force (Sagittal Push)
        # We must access env.envs[0].unwrapped to reach the Bullet core
        bullet_env = env.envs[0].unwrapped 

        if force_active and simtime < FORCE_DURATION + TIME_BEFORE_FORCE:
            if is_in_force_window:
                # Linear Ramp-up
                elapsed_push = simtime - TIME_BEFORE_FORCE
                current_scale = min(elapsed_push / RAMP_DURATION, 1.0)
                current_force = force_magnitude * current_scale
                
                bullet_env.set_bullet_action({
                    "external_forces": {
                        "base": {
                            "force": [current_force, 0.0, 0.0],  # push forward 
                            "position" : [0.0, 0.0, 0.0]        # push at the center
                        }
                    }
                })
        elif force_active:
            # Stop applying force after duration
            bullet_env.set_bullet_action({})
            force_active = False

        # 3. Step the environment
        # VecEnv step returns: obs, reward, done, info
        observation, _, done, info = env.step(action)
        simtime += dt

        # 4. Check for Fall
        # VecEnv observations are shape (1, 12), we want the first row
        latest_obs = observation[0]
        
        # In NewWrapper, Index 0 is Pitch
        pitch = latest_obs[0] 
        pitches.append(pitch)

        # Failure condition (Fall detected)
        if abs(pitch) > 1.0: # ~57 degrees
            success = False
            break

        # In VecEnv, 'done' is an array of booleans
        if done[0]:
            # If it reset automatically, check why. 
            # If pitch was high, it failed. If time ran out, it might be success.
            # But usually we break manually on pitch before this.
            if abs(pitch) > 1.0:
                success = False
            break
            
        # Success condition: Survived the force + stabilization time
        if not force_active and simtime > FORCE_DURATION + TIME_AFTER_FORCE + TIME_BEFORE_FORCE:
            success = True
            break

    return success

def main(policy_path: str, training: bool) -> None:
    env_settings = EnvSettings()
    init_state = None
    if training:
        training_settings = TrainingSettings()
        init_state = RobotState(
            randomization=RobotStateRandomization(
                **training_settings.init_rand
            ),
        )

    # --- 1. Create Base Environment (UpkieServos) ---
    # We use the raw class, not gym.make, to ensure we control the wrapping
    base_env = UpkieServos(
        frequency=env_settings.agent_frequency,
        init_state=init_state,
        max_ground_velocity=env_settings.max_ground_velocity,
        regulate_frequency=True, # Real-time regulation for visualization
        spine_config=env_settings.spine_config,
    )

    # --- 2. Apply Your New Wrapper ---
    env = NewWrapper(base_env)

    # --- 3. Vectorize & Normalize ---
    # We must replicate the training architecture (VecNormalize)
    # norm_reward=False because we don't care about rewards during testing
    # training=False stops it from updating statistics (uses default or loaded ones)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    try:
        # Load the PPO Policy
        ppo_settings = PPOSettings()
        policy = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={
                "net_arch": {
                    "pi": ppo_settings.net_arch_pi,
                    "vf": ppo_settings.net_arch_vf,
                },
            },
            verbose=0,
        )
        policy.set_parameters(policy_path)

        # --- MAIN TEST LOOP (1N to 20N) ---
        print("Starting Sagittal Force Test (1N to 20N)...")
        results = []
        
        # Range from 1 to 20 (inclusive)
        for force in range(1, 21):
            print(f"\nTesting Force: {force} N...")
            
            # Run the episode
            is_success = run_episode(env, policy, float(force))
            
            # Log result
            status = "Success" if is_success else "Failure"
            print(f"Result: {force} N -> {status}")
            results.append((force, is_success))
            
            # Optional: Stop testing if the robot fails to save time
            if not is_success:
                 print("Robot failed. Stopping tests.")
                 break
        
        print("\n--- Final Summary ---")
        for f, s in results:
            print(f"Force {f:2} N : {'[PASS]' if s else '[FAIL]'}")
            
    finally:
        env.close()

if __name__ == "__main__":
    if on_raspi():
        configure_agent_process()

    args = parse_command_line_arguments()

    # Policy parameters
    policy_path = args.policy
    if policy_path is None:
        script_dir = Path(__file__).parent.resolve()
        policy_dir = script_dir.parent / "policy"
        policy_path = policy_dir / "params.zip"
    if str(policy_path).endswith(".zip"):
        policy_path = str(policy_path)[:-4]
    logging.info("Loading policy from %s.zip", policy_path)

    # Configuration
    config_path = Path(policy_path).parent / "operative_config.gin"
    logging.info("Loading policy configuration from %s", config_path)
    try:
        gin.parse_config_file(str(config_path))
    except:
        logging.info("Could not load config file, using defaults.")

    try:
        main(policy_path, args.training)
    except KeyboardInterrupt:
        logging.info("Caught a keyboard interrupt")