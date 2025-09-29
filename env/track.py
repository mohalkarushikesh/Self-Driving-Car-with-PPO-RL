import pygame
import os
import numpy as np

class Track:
    def __init__(self, track_path, width, height, headless=False):
        self.headless = headless
        
        # Initialize pygame for image loading (even in headless mode)
        if not pygame.get_init():
            if headless:
                # Set dummy video driver for headless mode
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.init()
            
        # Only create display surface if not headless
        if not self.headless and pygame.display.get_surface() is None:
            pygame.display.set_mode((width, height))
        
        # Resolve track path - check if it's a relative path and look in images folder
        if not os.path.isabs(track_path):
            # If it's a relative path, try to find it in the images directory
            images_dir = os.path.join(os.path.dirname(__file__), "..", "images")
            full_path = os.path.join(images_dir, track_path)
            
            if os.path.exists(full_path):
                track_path = full_path
            elif os.path.exists(track_path):
                # Use the original path if it exists
                pass
            else:
                # Try common track names in images directory
                common_names = ["track.png", "track2.jpg", "track3.jpg"]
                for name in common_names:
                    test_path = os.path.join(images_dir, name)
                    if os.path.exists(test_path):
                        track_path = test_path
                        break
                else:
                    raise FileNotFoundError(f"Track image not found. Tried: {full_path}, {track_path}, and common names in {images_dir}")
        
        # Load track image - handle conversion properly
        try:
            # Load the image first
            self.original_image = pygame.image.load(track_path)
            
            # Only convert if we have a display surface (not in pure headless mode)
            if not headless and pygame.display.get_surface() is not None:
                self.original_image = self.original_image.convert()
            else:
                # In headless mode, convert to a format that doesn't require display
                self.original_image = self.original_image.convert_alpha()
                
        except pygame.error as e:
            if "No video mode has been set" in str(e):
                # Fallback: create a dummy display surface for conversion
                if headless:
                    os.environ['SDL_VIDEODRIVER'] = 'dummy'
                    pygame.init()
                    pygame.display.set_mode((1, 1))  # Minimal display
                    self.original_image = pygame.image.load(track_path).convert()
                else:
                    raise e
            else:
                raise e
        
        self.width = width
        self.height = height
        
        # Scale image to fit window
        self.image = pygame.transform.scale(self.original_image, (width, height))
        
        # Create collision mask for better collision detection
        self.mask = pygame.mask.from_surface(self.image)
        
        # Define track colors (adjust these based on your track image)
        # These are the colors that represent the track (road)
        self.track_colors = [
            (255, 255, 255),  # White
            (200, 200, 200),  # Light gray
            (150, 150, 150),  # Gray
            (100, 100, 100),  # Dark gray
            (50, 50, 50),     # Very dark gray
            (0, 0, 0),        # Black (if track is black)
        ]
        
        # Create a more sophisticated track detection
        self._create_track_mask()
        
    def _create_track_mask(self):
        """Create a mask that identifies track pixels more accurately"""
        # Create a new surface for the track mask
        self.track_mask = pygame.Surface((self.width, self.height))
        self.track_mask.fill((0, 0, 0))  # Start with black (not track)
        
        # Get the image data
        for x in range(self.width):
            for y in range(self.height):
                pixel_color = self.image.get_at((x, y))
                
                # Check if pixel is part of the track
                if self._is_track_pixel(pixel_color):
                    self.track_mask.set_at((x, y), (255, 255, 255))  # White = track
        
        # Create mask from the track surface
        self.track_collision_mask = pygame.mask.from_surface(self.track_mask)
    
    def _is_track_pixel(self, pixel_color):
        """Determine if a pixel is part of the track"""
        # Convert to RGB if needed
        if len(pixel_color) == 4:
            r, g, b, a = pixel_color
        else:
            r, g, b = pixel_color
        
        # Check against track colors with tolerance
        for track_color in self.track_colors:
            if len(track_color) == 3:
                tr, tg, tb = track_color
            else:
                tr, tg, tb, ta = track_color
            
            # Calculate color distance
            color_distance = ((r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2) ** 0.5
            
            # If color is close enough to track color, consider it track
            if color_distance < 50:  # Tolerance for color matching
                return True
        
        return False
    
    def is_off_track(self, car):
        """Check if car is off the track using improved detection"""
        # Get car position
        car_x = int(car.x)
        car_y = int(car.y)
        
        # Check bounds
        if car_x < 0 or car_x >= self.width or car_y < 0 or car_y >= self.height:
            return True
        
        # Check if car is on track using the improved mask
        try:
            # Get the pixel at car position
            pixel_color = self.track_mask.get_at((car_x, car_y))
            
            # If pixel is black (0, 0, 0), car is off track
            if pixel_color == (0, 0, 0):
                return True
                
        except IndexError:
            # If we can't get the pixel, assume off track
            return True
        
        return False
    
    def get_track_distance(self, x, y, angle, max_distance=200):
        """Get distance to track boundary in a given direction"""
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate direction vector
        dx = np.sin(angle_rad)
        dy = -np.cos(angle_rad)
        
        # Start from car position
        current_x = x
        current_y = y
        
        # Step along the ray
        for distance in range(1, max_distance + 1):
            # Calculate next position
            current_x = x + dx * distance
            current_y = y + dy * distance
            
            # Check bounds
            if (current_x < 0 or current_x >= self.width or 
                current_y < 0 or current_y >= self.height):
                return distance / max_distance
            
            # Check if we hit track boundary
            try:
                pixel_color = self.track_mask.get_at((int(current_x), int(current_y)))
                if pixel_color == (0, 0, 0):  # Hit non-track pixel
                    return distance / max_distance
            except IndexError:
                return distance / max_distance
        
        return 1.0  # Max distance reached
    
    def draw(self, surface):
        """Draw the track"""
        if not self.headless:
            surface.blit(self.image, (0, 0))
            
            # Optional: Draw track mask for debugging
            # surface.blit(self.track_mask, (0, 0))
