import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from env.car import Car
from env.track import Track
import random
import os

# Initialize pygame globally to avoid multiple init calls
# pygame.init()
# pygame.display.set_mode((1500, 800))

class SelfDrivingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, window_size=(1500, 800), track_path="track.png", 
                 continuous=False, visualize_sensors=False, domain_randomization=False, 
                 enhanced_observations=True):
        super().__init__()
        self.render_mode = render_mode
        self.window_width, self.window_height = window_size
        
        # Resolve track path to look in images directory
        if not os.path.isabs(track_path):
            images_dir = os.path.join(os.path.dirname(__file__), "..", "images")
            full_path = os.path.join(images_dir, track_path)
            if os.path.exists(full_path):
                self.track_path = full_path
            else:
                self.track_path = track_path
        else:
            self.track_path = track_path
            
        self.continuous = continuous
        self.visualize_sensors = visualize_sensors
        self.domain_randomization = domain_randomization
        self.enhanced_observations = enhanced_observations

        # Initialize pygame early to avoid issues
        self._pygame_initialized = False
        if not pygame.get_init():
            if self.render_mode != "human":
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.init()
            self._pygame_initialized = True

        if self.continuous:
            # throttle [-1,1], steer [-1,1], brake [0,1]
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
                                           high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(7)

        # Enhanced observation space with more information
        if self.enhanced_observations:
            # 5 ray distances + speed + angle + position + angular_velocity + previous_action + progress + corner_info
            # 5 + 1 + 2 + 2 + 1 + 3 + 1 + 2 = 17 dimensions
            low = np.array([0.0] * 5 + [0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            high = np.array([1.0] * 5 + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        else:
            # Original observation space for backward compatibility
            # 5 ray distances + speed + angle + position = 10 dimensions
            low = np.array([0.0] * 5 + [0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
            high = np.array([1.0] * 5 + [1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.win = None
        self.clock = None
        self.track = None
        self.car = None
        
        self.start_x = self.window_width // 2
        self.start_y = int(self.window_height * 0.85)

        self.steps = 0
        self.max_episode_steps = 2000
        
        # Track progress and previous states for better rewards
        self.previous_action = np.array([0.0, 0.0, 0.0]) if continuous else 0
        self.previous_position = (self.start_x, self.start_y)
        self.total_distance_traveled = 0.0
        self.straight_line_distance = 0.0
        self.last_progress_check = 0
        
        # Track following variables
        self.last_position = (self.start_x, self.start_y)
        self.stuck_counter = 0
        self.stuck_threshold = 50  # Steps before considering stuck
        self.min_movement_threshold = 5.0  # Minimum pixels to move per step
        self.last_angle = 0.0

    def _init_pygame(self):
        """Initialize pygame only when needed for rendering"""
        if not self._pygame_initialized:
            if self.render_mode != "human":
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.init()
            self._pygame_initialized = True

    def _get_obs(self):
        rays = self.car.sense(self.track)
        speed_norm = np.clip(abs(self.car.speed) / max(self.car.max_speed, 1e-6), 0.0, 1.0)
        ang_rad = np.radians(self.car.angle)
        posx = np.clip(self.car.x / self.window_width, 0.0, 1.0)
        posy = np.clip(self.car.y / self.window_height, 0.0, 1.0)
        
        if self.enhanced_observations:
            # Enhanced observations with additional information
            angular_vel_norm = np.clip(self.car.angular_velocity / 10.0, -1.0, 1.0)
            
            # Add previous action for temporal consistency
            if self.continuous:
                prev_action = self.previous_action  # 3 dimensions
            else:
                prev_action = np.array([float(self.previous_action)])  # 1 dimension
            
            # Add progress information
            progress = self._calculate_progress()
            
            # Add corner detection information
            corner_info = self._detect_corner_approach(rays)
            
            # Build observation: 5 rays + speed + angle + position + angular_vel + prev_action + progress + corner_info
            obs = rays + [speed_norm, np.sin(ang_rad), np.cos(ang_rad), 
                         posx, posy, angular_vel_norm] + prev_action.tolist() + [progress] + corner_info
            
            return np.array(obs, dtype=np.float32)
        else:
            # Original observation space for backward compatibility
            # 5 rays + speed + angle + position = 10 dimensions
            return np.array(rays + [speed_norm, np.sin(ang_rad), np.cos(ang_rad), posx, posy], dtype=np.float32)

    def _get_info(self):
        info = {
            "speed": self.car.speed, 
            "x": self.car.x, 
            "y": self.car.y
        }
        
        if self.enhanced_observations:
            info.update({
                "progress": self._calculate_progress(),
                "distance_traveled": self.total_distance_traveled,
                "stuck_counter": self.stuck_counter
            })
            
        return info

    def _detect_corner_approach(self, rays):
        """Detect if approaching a corner and return corner information"""
        # Check if we're approaching a corner based on ray patterns
        left_rays = rays[:2]
        right_rays = rays[3:]
        center_ray = rays[2]
        
        # Detect sharp turns by looking at asymmetry in side rays
        left_clearance = np.mean(left_rays)
        right_clearance = np.mean(right_rays)
        asymmetry = abs(left_clearance - right_clearance)
        
        # Detect if we're in a corner (low clearance on one side)
        in_corner = min(left_clearance, right_clearance) < 0.4
        
        # Detect corner approach (reducing clearance ahead)
        corner_approaching = center_ray < 0.6 and asymmetry > 0.3
        
        return [float(in_corner), float(corner_approaching)]

    def _calculate_progress(self):
        """Calculate how far the car has progressed along the track"""
        # Simple progress calculation based on Y position (assuming track goes downward)
        start_y = int(self.window_height * 0.85)
        current_progress = max(0, (start_y - self.car.y) / start_y)
        return np.clip(current_progress, 0.0, 1.0)

    def _calculate_forward_movement_reward(self):
        """Reward for moving forward along the track in any direction"""
        # Calculate movement in the direction the car is facing
        current_pos = (self.car.x, self.car.y)
        last_pos = self.last_position
        
        # Calculate movement vector
        dx = current_pos[0] - last_pos[0]
        dy = current_pos[1] - last_pos[1]
        movement_distance = np.sqrt(dx**2 + dy**2)
        
        if movement_distance < 0.1:  # Very small movement
            return -0.01  # Small penalty for not moving
        
        # Calculate the direction the car is facing
        car_angle_rad = np.radians(self.car.angle)
        forward_dx = np.sin(car_angle_rad)
        forward_dy = -np.cos(car_angle_rad)
        
        # Calculate how much of the movement is in the forward direction
        forward_component = (dx * forward_dx + dy * forward_dy) / movement_distance
        
        if forward_component > 0.5:  # Moving mostly forward
            return movement_distance * 0.1  # Strong reward for forward movement
        elif forward_component > 0:  # Moving somewhat forward
            return movement_distance * 0.05  # Moderate reward
        elif forward_component > -0.5:  # Moving sideways
            return movement_distance * 0.02  # Small reward for any movement
        else:  # Moving backward
            return movement_distance * -0.05  # Penalty for backward movement

    def _calculate_speed_reward(self):
        """Reward for maintaining good speed"""
        speed_ratio = abs(self.car.speed) / max(self.car.max_speed, 1e-6)
        
        # Reward moderate to high speed
        if speed_ratio > 0.3:
            return speed_ratio * 0.05  # Reward for good speed
        else:
            return -0.02  # Small penalty for being too slow

    def _calculate_center_reward(self, rays):
        """Reward for staying centered on the track"""
        center_ray = rays[2]
        left_rays = rays[:2]
        right_rays = rays[3:]
        
        # Reward being centered
        center_reward = center_ray * 0.1
        
        # Reward balanced left/right clearance
        left_clearance = np.mean(left_rays)
        right_clearance = np.mean(right_rays)
        balance_reward = (1.0 - abs(left_clearance - right_clearance)) * 0.05
        
        return center_reward + balance_reward

    def _calculate_stuck_penalty(self):
        """Penalty for getting stuck in the same position"""
        if self.stuck_counter > self.stuck_threshold:
            return -0.1  # Penalty for being stuck
        return 0.0

    def _calculate_direction_reward(self):
        """Reward for moving in the right direction"""
        # Calculate forward movement based on angle and speed
        forward_movement = -np.cos(np.radians(self.car.angle)) * self.car.speed
        
        if forward_movement > 0:
            return 0.02  # Reward for moving forward
        else:
            return -0.01  # Small penalty for going backward

    def _calculate_smoothness_penalty(self, action):
        """Penalize erratic steering to reduce shaking"""
        if self.continuous:
            # Penalize large steering changes
            steering_change = abs(action[1] - self.previous_action[1])
            return -steering_change * 0.02  # Reduced penalty
        else:
            # For discrete actions, penalize rapid direction changes
            if hasattr(self, 'previous_action') and self.previous_action != 0:
                if (self.previous_action in [3, 5] and action in [4, 6]) or \
                   (self.previous_action in [4, 6] and action in [3, 5]):
                    return -0.05  # Reduced penalty
            return 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.track is None:
            self.track = Track(self.track_path, self.window_width, self.window_height, headless=(self.render_mode != "human"))
        self.car = Car(self.start_x, self.start_y, headless=(self.render_mode != "human"))
        
        # Fix car orientation - set to -90 degrees to make it face left (horizontal)
        self.car.angle = -90    
        
        self.steps = 0

        if self.domain_randomization:
            friction_scale = 0.8 + random.random() * 0.6
            grip_scale = 0.8 + random.random() * 0.6
            self.car.randomize_dynamics(friction_scale=friction_scale, grip_scale=grip_scale)

        # Only create display window if rendering
        if self.render_mode == "human" and self.win is None:
            if not self._pygame_initialized:
                self._init_pygame()
            self.win = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Self-Driving Env")
            self.clock = pygame.time.Clock()

        # Reset tracking variables
        self.previous_action = np.array([0.0, 0.0, 0.0]) if self.continuous else 0
        self.previous_position = (self.start_x, self.start_y)
        self.total_distance_traveled = 0.0
        self.straight_line_distance = 0.0
        self.last_progress_check = 0.0
        self.last_position = (self.start_x, self.start_y)
        self.stuck_counter = 0
        self.last_angle = 0.0
        
        # Fix car orientation again after all resets
        self.car.angle = -90

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        if self.continuous:
            throttle = float(action[0])
            steer = float(action[1])
            brake = float(action[2])
            self.car.step_continuous(throttle, steer, brake=brake, handbrake=0.0, track=self.track) 
        else:
            self.car.step_action(int(action), track=self.track)

        self.steps += 1

        # Calculate distance traveled
        current_pos = (self.car.x, self.car.y)
        distance_moved = np.sqrt((current_pos[0] - self.previous_position[0])**2 + 
                                (current_pos[1] - self.previous_position[1])**2)
        self.total_distance_traveled += distance_moved
        self.previous_position = current_pos

        # Check if car is stuck
        if distance_moved < self.min_movement_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        terminated = False
        rays = self.car.sense(self.track)
        
        # **IMPROVED REWARD FUNCTION** - Works in all directions
        base_reward = 0.01  # Small base reward for staying alive
        
        # **1. FORWARD MOVEMENT REWARD** - Works in any direction
        forward_reward = self._calculate_forward_movement_reward()
        
        # **2. SPEED REWARD** - Encourage movement
        speed_reward = self._calculate_speed_reward()
        
        # **3. CENTER REWARD** - Stay on track
        center_reward = self._calculate_center_reward(rays)
        
        # **4. DIRECTION REWARD** - Move in right direction
        direction_reward = self._calculate_direction_reward()
        
        # **5. STUCK PENALTY** - Prevent getting stuck
        stuck_penalty = self._calculate_stuck_penalty()
        
        # **6. SMOOTHNESS PENALTY** - Reduce shaking
        smoothness_penalty = self._calculate_smoothness_penalty(action)
        
        # **7. CLEARANCE PENALTY** - Avoid walls
        min_ray = min(rays)
        clearance_penalty = (1.0 - min_ray) * -0.1
        
        # **Combine rewards with emphasis on forward movement**
        reward = (base_reward + forward_reward + speed_reward + center_reward + 
                 direction_reward + stuck_penalty + smoothness_penalty + clearance_penalty)

        # **Check for termination**
        if self.track.is_off_track(self.car):
            reward = -1.0
            terminated = True

        # **Terminate if stuck for too long**
        if self.stuck_counter > self.stuck_threshold * 2:
            reward = -0.5
            terminated = True

        truncated = self.steps >= self.max_episode_steps
        
        # Store current action and position for next step
        if self.continuous:
            self.previous_action = np.array([throttle, steer, brake])
        else:
            self.previous_action = int(action)
        
        self.last_position = current_pos
        self.last_angle = self.car.angle

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if not self._pygame_initialized:
            self._init_pygame()

        if self.win is None:
            self.win = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()

        self.win.fill((0, 0, 0))
        self.track.draw(self.win)
        self.car.draw(self.win)

        if self.visualize_sensors:
            rays = self.car.sense(self.track)
            angles = [-45, -22.5, 0, 22.5, 45]
            for i, norm_dist in enumerate(rays):
                a = angles[i]
                world_angle = np.radians(self.car.angle + a)
                dx = np.sin(world_angle)
                dy = -np.cos(world_angle)
                length = int(120 * norm_dist)
                start = (int(self.car.x), int(self.car.y))
                end = (int(self.car.x + dx * length), int(self.car.y + dy * length))
                color = (0, 255, 0) if norm_dist > 0.8 else (255, 165, 0) if norm_dist > 0.4 else (255, 0, 0)
                pygame.draw.line(self.win, color, start, end, 2)

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.win is not None:
            pygame.display.quit()
            pygame.quit()
            self.win = None
            self.clock = None
