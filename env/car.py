import pygame
import math
import os
import numpy as np

class Car:
    def __init__(self, x, y, headless=False):
        self.headless = headless
        
        # Only initialize pygame if not headless AND pygame not already initialized
        if not self.headless and not pygame.get_init():
            pygame.init()
            if pygame.display.get_surface() is None:
                pygame.display.set_mode((1500, 800))
        
        # Load car image (even in headless mode for collision detection)
        images_dir = os.path.join(os.path.dirname(__file__), "..", "images")
        car_path = os.path.join(images_dir, "car.png")
        
        if os.path.exists(car_path):
            self.original_image = pygame.image.load(car_path).convert_alpha()
            # Scale the car to a reasonable size (larger than before)
            self.original_image = pygame.transform.scale(self.original_image, (100, 100))
        else:
            # Create a simple car if image doesn't exist - make it larger
            self.original_image = pygame.Surface((100, 100), pygame.SRCALPHA)
            pygame.draw.rect(self.original_image, (255, 0, 0), (0, 0, 100, 100))
        
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect()
        
        # Car physics
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.max_speed = 8
        self.acceleration = 0.2
        self.friction = 0.95
        self.turn_speed = 3
        self.angular_velocity = 0
        
        # Car dimensions for collision detection - updated to match image size
        self.width = 100
        self.height = 100
        
        # Sensor angles for ray casting
        self.sensor_angles = [-45, -22.5, 0, 22.5, 45]
        self.sensor_length = 200
        
    def sense(self, track):
        """Improved sensing that works better with complex tracks"""
        distances = []
        
        for angle_offset in self.sensor_angles:
            # Calculate sensor direction
            sensor_angle = self.angle + angle_offset
            distance = track.get_track_distance(self.x, self.y, sensor_angle, self.sensor_length)
            distances.append(distance)
        
        return distances
    
    def step_continuous(self, throttle, steer, brake=0.0, handbrake=0.0, track=None):
        """Step the car with continuous controls"""
        # Apply throttle
        if throttle > 0:
            self.speed += throttle * self.acceleration
        elif throttle < 0:
            self.speed += throttle * self.acceleration * 0.5  # Reverse is slower
        
        # Apply brake
        if brake > 0:
            self.speed *= (1 - brake * 0.3)
        
        # Apply handbrake
        if handbrake > 0:
            self.speed *= (1 - handbrake * 0.5)
        
        # Apply friction
        self.speed *= self.friction
        
        # Limit speed
        self.speed = max(-self.max_speed * 0.5, min(self.speed, self.max_speed))
        
        # Apply steering - allow steering even when stationary
        if abs(steer) > 0.01:  # Only apply steering if there's input
            # Steering is more responsive when moving, but still works when stationary
            steering_multiplier = max(0.3, abs(self.speed) / self.max_speed)  # Minimum 30% steering effectiveness
            self.angular_velocity = steer * self.turn_speed * steering_multiplier
        else:
            # Apply angular friction when not steering
            self.angular_velocity *= 0.8
        
        # Update angle
        self.angle += self.angular_velocity
        
        # Update position
        angle_rad = math.radians(self.angle)
        self.x += math.sin(angle_rad) * self.speed
        self.y -= math.cos(angle_rad) * self.speed
        
        # Keep car in bounds
        self.x = max(0, min(self.x, 1500))
        self.y = max(0, min(self.y, 800))
    
    def step_action(self, action, track=None):
        """Step the car with discrete actions"""
        if action == 0:  # Do nothing
            pass
        elif action == 1:  # Accelerate
            self.speed += self.acceleration
        elif action == 2:  # Brake
            self.speed -= self.acceleration
        elif action == 3:  # Turn left
            self.angular_velocity = -self.turn_speed
        elif action == 4:  # Turn right
            self.angular_velocity = self.turn_speed
        elif action == 5:  # Accelerate + Turn left
            self.speed += self.acceleration
            self.angular_velocity = -self.turn_speed
        elif action == 6:  # Accelerate + Turn right
            self.speed += self.acceleration
            self.angular_velocity = self.turn_speed
        
        # Apply friction
        self.speed *= self.friction
        
        # Limit speed
        self.speed = max(-self.max_speed * 0.5, min(self.speed, self.max_speed))
        
        # Update angle
        self.angle += self.angular_velocity
        
        # Update position
        angle_rad = math.radians(self.angle)
        self.x += math.sin(angle_rad) * self.speed
        self.y -= math.cos(angle_rad) * self.speed
        
        # Keep car in bounds
        self.x = max(0, min(self.x, 1500))
        self.y = max(0, min(self.y, 800))
    
    def draw(self, surface):
        """Draw the car"""
        if not self.headless:
            # Rotate car image
            rotated_image = pygame.transform.rotate(self.original_image, -self.angle)
            rect = rotated_image.get_rect(center=(self.x, self.y))
            surface.blit(rotated_image, rect)
    
    def randomize_dynamics(self, friction_scale=1.0, grip_scale=1.0):
        """Randomize car dynamics for domain randomization"""
        self.friction *= friction_scale
        self.turn_speed *= grip_scale
