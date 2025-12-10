import gymnasium
import numpy as np
from gymnasium import spaces

class UpkieServosWrapper(gymnasium.Wrapper):
    """
    Wrapper for UpkieServos-v5 to support the 'All Robot Joints' task.
    Maps vector actions to the dictionary structure Upkie expects.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # --- Observation Space ---
        # We need: Joint Pos (6) + Joint Vel (6) + Pitch (1) + Pitch Deriv (1) = 14 dims
        # Order: [left_hip, left_knee, left_wheel, right_hip, right_knee, right_wheel]
        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(14,),
            dtype=np.float32
        )

        # --- Action Space ---
        # The exam requires: 2x hips, 2x knees, 1x ground velocity
        # Vector: [left_hip_pos, left_knee_pos, right_hip_pos, right_knee_pos, ground_vel]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )
        
        # Action Scaling
        self.max_leg_pos = 1.0   # +/- 1.0 radian range for legs
        self.max_wheel_vel = 2.0 # +/- 2.0 m/s for wheels

    def _dict_to_vec(self, obs_dict, pitch, pitch_vel):
        """Convert dictionary observation to flat vector."""
        q_pos = []
        q_vel = []
        
        # Standard Upkie joint names
        joint_names = [
            'left_hip', 'left_knee', 'left_wheel', 
            'right_hip', 'right_knee', 'right_wheel'
        ]
        
        for joint in joint_names:
            j_data = obs_dict.get(joint, {'position': 0.0, 'velocity': 0.0})
            
            # Force conversion to scalar float
            pos_val = float(j_data['position'])
            vel_val = float(j_data['velocity'])
            
            q_pos.append(pos_val)
            q_vel.append(vel_val)

        return np.concatenate([q_pos, q_vel, [pitch], [pitch_vel]], dtype=np.float32)

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        
        spine_obs = info.get('spine_observation', {})
        pitch = spine_obs.get('base_orientation', {}).get('pitch', 0.0)
        
        return self._dict_to_vec(obs_dict, pitch, 0.0), info

    def step(self, action):
        l_hip, l_knee, r_hip, r_knee, g_vel = action
        
        action_dict = {
            "left_hip": {"position": l_hip * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            "left_knee": {"position": l_knee * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            "right_hip": {"position": r_hip * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            "right_knee": {"position": r_knee * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            
            "left_wheel": {"position": float("nan"), "velocity": g_vel * self.max_wheel_vel, "kp": 0.0, "kd": 1.0},
            "right_wheel": {"position": float("nan"), "velocity": -g_vel * self.max_wheel_vel, "kp": 0.0, "kd": 1.0},
        }
        
        obs_dict, reward, terminated, truncated, info = self.env.step(action_dict)
        
        spine_obs = info.get('spine_observation', {})
        pitch = spine_obs.get('base_orientation', {}).get('pitch', 0.0)
        imu = spine_obs.get('imu', {})
        ang_vel = imu.get('angular_velocity', [0.0, 0.0, 0.0])
        pitch_vel = float(ang_vel[1])
        
        vector_obs = self._dict_to_vec(obs_dict, pitch, pitch_vel)
        
        return vector_obs, reward, terminated, truncated, info

class ServosReward(gymnasium.Wrapper):
    """
    Computes reward for balancing with Servos and provides 
    attributes for logging (fixing the AttributeError).
    """
    def __init__(self, env):
        super().__init__(env)
        # Initialize attributes required by RewardCallback
        self.last_position_reward = 0.0
        self.last_velocity_penalty = 0.0
        self.last_action_change_penalty = 0.0
        self.last_action = None

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Extract Pitch (Index 12 in UpkieServosWrapper obs)
        # Obs layout: 6 pos, 6 vel, pitch, pitch_vel
        pitch = obs[12]
        
        # 1. Position Reward: Keep pitch close to 0
        position_reward = np.exp(- (pitch / 0.2)**2)
        
        # 2. Velocity Penalty: Penalize high wheel speeds (Indices 8 and 11)
        # q_vel indices: 6=l_hip, 7=l_knee, 8=l_wheel, 9=r_hip, 10=r_knee, 11=r_wheel
        l_wheel_vel = obs[8]
        r_wheel_vel = obs[11]
        velocity_penalty = 0.01 * (l_wheel_vel**2 + r_wheel_vel**2)
        
        # 3. Action Smoothness
        if self.last_action is not None:
            action_change = np.sum(np.abs(action - self.last_action))
            action_change_penalty = 0.01 * action_change
        else:
            action_change_penalty = 0.0
        self.last_action = action.copy()

        # Total Reward
        reward = position_reward - velocity_penalty - action_change_penalty
        
        # Update attributes for logger
        self.last_position_reward = position_reward
        self.last_velocity_penalty = velocity_penalty
        self.last_action_change_penalty = action_change_penalty
        
        return obs, reward, terminated, truncated, info

def wrap_servos_env(env, env_settings):
    # Apply your robust wrappers here too (Noise, etc.)
    # Note: DefineReward might need updates if it expects different obs structure
    env = UpkieServosWrapper(env)
    return env
