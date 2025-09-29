#!/usr/bin/env python3
"""
Quick demo script for visual testing of trained models.
"""

import os
import argparse
import numpy as np
import pygame
from stable_baselines3 import PPO
from env.gym_env import SelfDrivingEnv

def demo_model(model_path, continuous=None, visualize_sensors=True, num_episodes=1, enhanced=True):
    """
    Quick visual demo of a trained model.
    """
    print(f"🎬 Demo: {model_path}")
    
    # Load model first to detect action space
    model = PPO.load(model_path)
    
    # Auto-detect action space from model
    if continuous is None:
        # Check if model was trained with continuous actions
        if hasattr(model.action_space, 'shape') and len(model.action_space.shape) > 0:
            continuous = True
        else:
            continuous = False
    
    print(f"🔧 Action space: {'Continuous' if continuous else 'Discrete'}")
    print(f"🔧 Enhanced observations: {'Yes' if enhanced else 'No'}")
    
    # Create environment with same settings as training
    env = SelfDrivingEnv(
        render_mode="human",
        continuous=continuous,
        visualize_sensors=visualize_sensors,
        enhanced_observations=enhanced  # Use enhanced observations to match training
    )
    
    print(f"🎮 Running {num_episodes} episode(s)...")
    
    for episode in range(num_episodes):
        print(f"\n🎮 Episode {episode + 1}")
        
        try:
            obs, info = env.reset()
            total_reward = 0
            step = 0
            
            while True:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                
                # Render
                env.render()
                
                # Handle pygame events to prevent freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return
                
                # Check if episode is done
                if terminated or truncated:
                    break
                
                # Limit steps to prevent infinite loops
                if step > 2000:
                    break
            
            print(f"✅ Episode {episode + 1} completed!")
            print(f"   Steps: {step}")
            print(f"   Total reward: {total_reward:.2f}")
            print(f"   Progress: {info.get('progress', 0):.2f}")
            
        except Exception as e:
            print(f"❌ Error during demo: {e}")
            break
    
    env.close()
    print("\n🎉 Demo completed!")

def main():
    parser = argparse.ArgumentParser(description="Demo trained self-driving car model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--continuous", action="store_true", help="Use continuous actions")
    parser.add_argument("--discrete", action="store_true", help="Use discrete actions")
    parser.add_argument("--no-sensors", action="store_true", help="Don't visualize sensors")
    parser.add_argument("--no-enhanced", action="store_true", help="Don't use enhanced observations")
    
    args = parser.parse_args()
    
    # Determine action space
    continuous = None
    if args.continuous:
        continuous = True
    elif args.discrete:
        continuous = False
    
    # Determine enhanced observations
    enhanced = not args.no_enhanced
    
    demo_model(
        args.model,
        continuous=continuous,
        visualize_sensors=not args.no_sensors,
        num_episodes=args.episodes,
        enhanced=enhanced
    )

if __name__ == "__main__":
    main()
