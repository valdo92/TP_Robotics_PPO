import gymnasium
import numpy as np
from gymnasium import spaces
from upkie.envs import UpkieServos

class UpkieServosWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # --- 1. Define Observation Space  ---
        # We need: Joint Pos (6) + Joint Vel (6) + Pitch (1) + Pitch Deriv (1) = 14 dims
        # (Assuming 6 joints: 2 hips, 2 knees, 2 wheels)
        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(14,),
            dtype=np.float32
        )

        # --- 2. Define Action Space [cite: 54] ---
        # 2 hips (pos), 2 knees (pos), 1 ground_velocity (vel) = 5 dims
        # Range: -1 to 1 (Network output)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )
        
        # Scale factors (adjust these based on your robot's limits)
        self.max_leg_pos = 1.0  # radians (approx 57 deg)
        self.max_wheel_vel = 1.0 # m/s (or rad/s depending on config)

    def parse_first_observation(self, obs_dict):
        # Helper to process the initial observation after reset
        # We need an empty info dict or mock data for the first step
        return self._dict_to_vec(obs_dict, pitch=0.0, pitch_vel=0.0)

    def _dict_to_vec(self, obs_dict, pitch, pitch_vel):
        # Extract dictionary values to a flat vector
        # Note: Verify key names in obs_dict usually match servo names
        # Standard Upkie joints: left_hip, left_knee, left_wheel, right_hip, right_knee, right_wheel
        
        q_pos = []
        q_vel = []
        
        # Order matters! Let's keep: Left Leg, Right Leg
        # (check upkie config for exact naming, usually: left_hip, left_knee, left_wheel...)
        joint_names = [
            'left_hip', 'left_knee', 'left_wheel',
            'right_hip', 'right_knee', 'right_wheel'
        ]
        
        for joint in joint_names:
            # Safely get data, assuming structure obs_dict[joint]['position']
            # If structure is flattened, adjust accordingly. 
            # UpkieServos typically returns nested dicts.
            if joint in obs_dict:
                q_pos.append(obs_dict[joint]['position'])
                q_vel.append(obs_dict[joint]['velocity'])
            else:
                q_pos.append(0.0)
                q_vel.append(0.0)

        # Combine into vector: [Positions (6), Velocities (6), Pitch, Pitch_Vel]
        return np.concatenate([q_pos, q_vel, [pitch], [pitch_vel]], dtype=np.float32)

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        
        #  Pitch is in info['spine_observation']
        spine_obs = info.get('spine_observation', {})
        pitch = spine_obs.get('base_orientation', {}).get('pitch', 0.0)
        # We might not have derivative on first step, assume 0
        
        return self._dict_to_vec(obs_dict, pitch, 0.0), info

    def step(self, action):
        # --- 3. Map Action (5 dims) to Dictionary ---
        # Action vector: [left_hip, left_knee, right_hip, right_knee, ground_vel]
        
        # Rescale actions if necessary (policy outputs [-1, 1])
        l_hip, l_knee, r_hip, r_knee, g_vel = action
        
        action_dict = {
            "left_hip": {"position": l_hip * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            "left_knee": {"position": l_knee * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            "right_hip": {"position": r_hip * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            "right_knee": {"position": r_knee * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            
            # Wheels are velocity controlled
            "left_wheel": {"position": float("nan"), "velocity": g_vel * self.max_wheel_vel, "kp": 0.0, "kd": 1.0},
            "right_wheel": {"position": float("nan"), "velocity": -g_vel * self.max_wheel_vel, "kp": 0.0, "kd": 1.0}, 
            # Note: One wheel is usually inverted in differential drive; check Upkie docs if +v goes forward for both.
        }
        
        obs_dict, reward, terminated, truncated, info = self.env.step(action_dict)
        
        # Extract pitch [cite: 51]
        spine_obs = info.get('spine_observation', {})
        pitch = spine_obs.get('base_orientation', {}).get('pitch', 0.0)
        # Use simple difference for derivative if not provided (or check imu dict)
        pitch_vel = spine_obs.get('imu', {}).get('angular_velocity', [0,0,0])[1] # y-axis
        
        vector_obs = self._dict_to_vec(obs_dict, pitch, pitch_vel)
        
        return vector_obs, reward, terminated, truncated, info

def wrap_servos_env(env, env_settings):
    # Apply your robust wrappers here too (Noise, etc.)
    # Note: DefineReward might need updates if it expects different obs structure
    env = UpkieServosWrapper(env)
    return env