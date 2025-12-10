import gymnasium as gym
import numpy as np
from upkie.envs.upkie_servos import UpkieServos
from upkie.utils.clamp import clamp_and_warn

class NewWrapper(gym.Wrapper):
    """
    UpkieServos wrapper that:
    1. Vectorizes observations (matches UpkieGroundVelocity order first).
    2. allows control of all leg joints + ground velocity (Question 4 / E1).
    """

    def __init__(self, env: UpkieServos, wheel_radius: float = 0.06, left_wheeled: bool = True):
        super().__init__(env)
        self.env = env
        self.wheel_radius = wheel_radius
        self.left_wheeled = left_wheeled

        # --- Action Space (Full Control) ---
        # Structure: [left_hip, left_knee, right_hip, right_knee, ground_velocity]
        # Size: 5
        
        # Get limits from the robot model directly for safety
        l_hip_low = self.env.action_space['left_hip']['position'].low[0]
        l_hip_high = self.env.action_space['left_hip']['position'].high[0]
        l_knee_low = self.env.action_space['left_knee']['position'].low[0]
        l_knee_high = self.env.action_space['left_knee']['position'].high[0]
        
        max_ground_vel = 2.0
        
        # 5 Dimensions now!
        self.action_space = gym.spaces.Box(
            low=np.array([l_hip_low, l_knee_low, l_hip_low, l_knee_low, -max_ground_vel], dtype=np.float32),
            high=np.array([l_hip_high, l_knee_high, l_hip_high, l_knee_high, max_ground_vel], dtype=np.float32),
            dtype=np.float32
        )

        # --- Observation Space ---
        # 0: Pitch
        # 1: Ground Position
        # 2: Pitch Rate
        # 3: Ground Velocity
        # 4-7: Joint Pos
        # 8-11: Joint Vel
        obs_dim = 12
        self.observation_space = gym.spaces.Box(
            low=-np.float32(np.inf), 
            high=np.float32(np.inf), 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        self.fall_pitch = 1.0  

        self.leg_joints = ['left_hip', 'left_knee', 'right_hip', 'right_knee']

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        obs = self._get_observation(info["spine_observation"])
        return obs, info

    def step(self, action):
        # 1. Unpack Action (Size 5)
        # Order must match action_space definition in __init__
        left_hip_cmd = action[0]
        left_knee_cmd = action[1]
        right_hip_cmd = action[2]
        right_knee_cmd = action[3]
        ground_vel_cmd = action[4]

        ground_vel_cmd = clamp_and_warn(
                    raw_ground_vel,
                    -2.0,  # Min limit
                    2.0,   # Max limit
                    label="ground_velocity"
                )
        # 2. Wheel Kinematics
        wheel_omega = ground_vel_cmd / self.wheel_radius
        left_wheel_sign = 1.0 if self.left_wheeled else -1.0
        left_wheel_vel = left_wheel_sign * wheel_omega
        right_wheel_vel = -left_wheel_vel

        # 3. Build Dictionary Action
        env_action = self.env.get_neutral_action()
        
        # Left Leg
        env_action['left_hip']['position'] = float(left_hip_cmd)
        env_action['left_knee']['position'] = float(left_knee_cmd)
        
        # Right Leg
        env_action['right_hip']['position'] = float(right_hip_cmd)
        env_action['right_knee']['position'] = float(right_knee_cmd)

        # Wheels
        env_action['left_wheel']['position'] = np.nan
        env_action['left_wheel']['velocity'] = float(left_wheel_vel)
        env_action['right_wheel']['position'] = np.nan
        env_action['right_wheel']['velocity'] = float(right_wheel_vel)

        # 4. Step
        # 2. Step the internal environment
        # Note: UpkieServos always returns terminated=False by default
        obs_dict, reward, terminated, truncated, info = self.env.step(env_action)
        
        # 3. Process Observation
        obs = self._get_observation(info['spine_observation'])

        # --- 4. NEW: FALL DETECTION ---
        # We extract pitch from the observation (Index 0 in our vector)
        pitch = obs[0] 
        
        # If pitch is too high, we force the episode to end
        if abs(pitch) > self.fall_pitch:
            terminated = True
            
        return obs, reward, terminated, truncated, info

    def _get_observation(self, spine_obs):
        base_orient = spine_obs["base_orientation"]
        pitch = base_orient["pitch"]
        pitch_rate = base_orient["angular_velocity"][1]
        
        ground_pos = spine_obs["wheel_odometry"]["position"]
        ground_vel = spine_obs["wheel_odometry"]["velocity"]

        # Leg Joints (order: LH, LK, RH, RK)
        joint_pos = []
        joint_vel = []
        for joint in self.leg_joints:
            joint_pos.append(spine_obs["servo"][joint]["position"])
            joint_vel.append(spine_obs["servo"][joint]["velocity"])

        obs_list = [pitch, ground_pos, pitch_rate, ground_vel] + joint_pos + joint_vel
        return np.array(obs_list, dtype=np.float32)