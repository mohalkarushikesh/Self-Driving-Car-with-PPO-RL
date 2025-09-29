import pygame
import sys
import os

# Add the parent directory to the path so we can import from env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.car import Car
from env.track import Track

def main_loop():
    pygame.init()
    screen = pygame.display.set_mode((1500, 800))
    pygame.display.set_caption("Self-Driving Car Environment")
    clock = pygame.time.Clock()
    
    # Create track and car
    track = Track("track.png", 1500, 800, headless=False)
    car = Car(750, 680, headless=False)  # Start position
    car.angle = -90 # Ensure car starts horizontally
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Handle keyboard input
        keys = pygame.key.get_pressed()
        
        # Convert keyboard input to continuous controls
        throttle = 0.0
        steer = 0.0
        brake = 0.0
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            throttle = -1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer = -1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer = 1.0
        if keys[pygame.K_SPACE]:
            brake = 1.0
        
        # Update car with continuous controls
        car.step_continuous(throttle, steer, brake, track=track)
        
        # Check if car is off track
        if track.is_off_track(car):
            print("Car went off track!")
            # Reset car position
            car.x = 750
            car.y = 680
            car.speed = 0
            car.angle = -90  # Make sure it faces left (horizontal)
        
        # Draw everything
        screen.fill((0, 0, 0))
        track.draw(screen)
        car.draw(screen)
        
        # Draw sensors
        rays = car.sense(track)
        angles = [-45, -22.5, 0, 22.5, 45]
        for i, norm_dist in enumerate(rays):
            a = angles[i]
            world_angle = pygame.math.Vector2(1, 0).rotate(car.angle + a)
            length = int(120 * norm_dist)
            start = (int(car.x), int(car.y))
            end = (int(car.x + world_angle.x * length), int(car.y + world_angle.y * length))
            color = (0, 255, 0) if norm_dist > 0.8 else (255, 165, 0) if norm_dist > 0.4 else (255, 0, 0)
            pygame.draw.line(screen, color, start, end, 2)
        
        # Display info
        font = pygame.font.Font(None, 36)
        speed_text = font.render(f"Speed: {car.speed:.2f}", True, (255, 255, 255))
        angle_text = font.render(f"Angle: {car.angle:.1f}°", True, (255, 255, 255))
        screen.blit(speed_text, (10, 10))
        screen.blit(angle_text, (10, 50))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main_loop()
