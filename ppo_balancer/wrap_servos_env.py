import gymnasium
import numpy as np
from gymnasium import spaces

# Define indices for code readability and safety
IDX_L_HIP_POS, IDX_L_HIP_VEL = 0, 6
IDX_L_KNEE_POS, IDX_L_KNEE_VEL = 1, 7
IDX_L_WHEEL_POS, IDX_L_WHEEL_VEL = 2, 8
IDX_R_HIP_POS, IDX_R_HIP_VEL = 3, 9
IDX_R_KNEE_POS, IDX_R_KNEE_VEL = 4, 10
IDX_R_WHEEL_POS, IDX_R_WHEEL_VEL = 5, 11
IDX_PITCH = 12
IDX_PITCH_VEL = 13

class UpkieServosWrapper(gymnasium.Wrapper):
    """
    Wrapper for UpkieServos-v5 to support the 'All Robot Joints' task.
    Maps vector actions to the dictionary structure Upkie expects.
    Also handles External Forces for Curriculum Learning.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # --- Observation Space ---
        # 14 dims: 6 Pos, 6 Vel, Pitch, Pitch Deriv
        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(14,),
            dtype=np.float32
        )

        # --- Action Space ---
        # 5 dims: [left_hip, left_knee, right_hip, right_knee, ground_vel]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )
        
        # Action Scaling
        self.max_leg_pos = 1.0   # Radians
        self.max_wheel_vel = 2.0 # m/s

        # Force Curriculum State
        self.current_max_force = 0.0
        self._force_timer = 0.0
        self._current_force_vec = np.zeros(3)
        
        # Config
        self.dt = 1.0 / 200.0 # Assuming 200Hz, adjust if different
        self.push_duration = 1.0
        self.push_probability = 0.02

    def set_max_force(self, max_force: float):
        """Called by the Curriculum Callback in train.py"""
        self.current_max_force = max_force

    def _dict_to_vec(self, obs_dict, pitch, pitch_vel):
        q_pos = []
        q_vel = []
        
        # Order MUST match the indices defined at top of file
        joint_names = [
            'left_hip', 'left_knee', 'left_wheel', 
            'right_hip', 'right_knee', 'right_wheel'
        ]
        
        for joint in joint_names:
            j_data = obs_dict.get(joint, {'position': 0.0, 'velocity': 0.0})
            q_pos.append(float(j_data['position']))
            q_vel.append(float(j_data['velocity']))

        return np.concatenate([q_pos, q_vel, [pitch], [pitch_vel]], dtype=np.float32)

    def _handle_force_logic(self):
        """Manage the random pushing logic."""
        if self._force_timer > 0:
            self._force_timer -= self.dt
            if self._force_timer <= 0:
                self._current_force_vec = np.zeros(3)
        
        elif self.current_max_force > 0.1 and np.random.random() < self.push_probability:
            self._force_timer = self.push_duration
            f_mag = np.random.uniform(self.current_max_force * 0.5, self.current_max_force)
            # Sagittal push (X-axis)
            direction = 1 if np.random.random() > 0.5 else -1
            self._current_force_vec = np.array([f_mag * direction, 0.0, 0.0])
            
        return self._current_force_vec.tolist()

    def reset(self, **kwargs):
        self._current_force_vec = np.zeros(3)
        self._force_timer = 0.0
        
        obs_dict, info = self.env.reset(**kwargs)
        spine_obs = info.get('spine_observation', {})
        pitch = spine_obs.get('base_orientation', {}).get('pitch', 0.0)
        return self._dict_to_vec(obs_dict, pitch, 0.0), info

    def step(self, action):
        l_hip, l_knee, r_hip, r_knee, g_vel = action
        
        # 1. Create the Servos Dictionary
        action_dict = {
            "left_hip": {"position": l_hip * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            "left_knee": {"position": l_knee * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            "right_hip": {"position": r_hip * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            "right_knee": {"position": r_knee * self.max_leg_pos, "velocity": 0.0, "kp": 10.0, "kd": 1.0},
            # Note: Check signs. If robot spins, flip one of these signs.
            "left_wheel": {"position": float("nan"), "velocity": g_vel * self.max_wheel_vel, "kp": 0.0, "kd": 1.0},
            "right_wheel": {"position": float("nan"), "velocity": -g_vel * self.max_wheel_vel, "kp": 0.0, "kd": 1.0},
        }
        
        # 2. Inject External Forces (Curriculum)
        force = self._handle_force_logic()
        if np.linalg.norm(force) > 0:
            action_dict["external_forces"] = {
                "base": {
                    "force": force,
                    "position": [0.0, 0.0, 0.0],
                }
            }
        
        # 3. Step
        obs_dict, reward, terminated, truncated, info = self.env.step(action_dict)
        
        # 4. Process Observation
        spine_obs = info.get('spine_observation', {})
        pitch = spine_obs.get('base_orientation', {}).get('pitch', 0.0)
        
        # Get Pitch Velocity from IMU angular velocity Y-axis (Body frame)
        imu = spine_obs.get('imu', {})
        ang_vel = imu.get('angular_velocity', [0.0, 0.0, 0.0])
        pitch_vel = float(ang_vel[1])
        
        vector_obs = self._dict_to_vec(obs_dict, pitch, pitch_vel)
        
        return vector_obs, reward, terminated, truncated, info

class ServosReward(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_position_reward = 0.0
        self.last_velocity_penalty = 0.0
        self.last_action_change_penalty = 0.0
        self.last_action = None

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Extract using constants (Robust)
        pitch = obs[IDX_PITCH]
        l_wheel_vel = obs[IDX_L_WHEEL_VEL]
        r_wheel_vel = obs[IDX_R_WHEEL_VEL]
        
        # 1. Position Reward (Stay Upright)
        position_reward = np.exp(- (pitch / 0.2)**2)
        
        # 2. Velocity Penalty (Don't race off)
        velocity_penalty = 0.01 * (l_wheel_vel**2 + r_wheel_vel**2)
        
        # 3. Action Smoothness
        if self.last_action is not None:
            action_change = np.sum(np.abs(action - self.last_action))
            action_change_penalty = 0.005 * action_change # Reduced slightly
        else:
            action_change_penalty = 0.0
        self.last_action = action.copy()

        # Total
        reward = position_reward - velocity_penalty - action_change_penalty
        
        self.last_position_reward = position_reward
        self.last_velocity_penalty = velocity_penalty
        self.last_action_change_penalty = action_change_penalty
        
        return obs, reward, terminated, truncated, info