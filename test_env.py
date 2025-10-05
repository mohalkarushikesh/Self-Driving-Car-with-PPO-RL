#!/usr/bin/env python3
"""
Test script to verify environment works correctly
"""

import numpy as np
from env.gym_env import SelfDrivingEnv

def test_environment():
    print("🧪 Testing Environment...")
    
    # Test environment creation
    env = SelfDrivingEnv(
        render_mode=None,
        continuous=True,
        enhanced_observations=True
    )
    
    print("✅ Environment created successfully")
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful - Observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Done={terminated}")
        
        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()
    
    env.close()
    print("✅ Environment test completed successfully!")

if __name__ == "__main__":
    test_environment() 
