#!/usr/bin/env python3
"""
Training script for self-driving car with PPO
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from env.gym_env import SelfDrivingEnv

def make_env(continuous=True, enhanced_observations=True):
    """Create environment factory"""
    def _thunk():
        return SelfDrivingEnv(
            render_mode=None,  # Headless training
            continuous=continuous,
            visualize_sensors=False,
            domain_randomization=False,
            enhanced_observations=enhanced_observations
        )
    return _thunk

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for self-driving car")
    parser.add_argument("--continuous", action="store_true", help="Use continuous action space")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced observations")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps per update")
    
    args = parser.parse_args()
    
    print("🚗 Self-Driving Car Training")
    print("=" * 50)
    print(f"Action space: {'Continuous' if args.continuous else 'Discrete'}")
    print(f"Enhanced observations: {'Yes' if args.enhanced else 'No'}")
    print(f"Total timesteps: {args.timesteps}")
    print("=" * 50)
    
    # Create environment
    env = DummyVecEnv([make_env(args.continuous, args.enhanced)])
    
    # Create model with better parameters for learning
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])  # Larger network
        )
    )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(args.continuous, args.enhanced)])
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=5000,  # Evaluate every 5000 steps
        deterministic=True,
        render=False
    )
    
    # Train the model
    print("🚀 Starting training...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model_name = f"ppo_self_driving_{'continuous' if args.continuous else 'discrete'}"
    if args.enhanced:
        model_name += "_enhanced"
    
    model_path = f"models/{model_name}"
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Test the model
    print("\n🎮 Testing trained model...")
    obs = env.reset()
    for i in range(3):
        print(f"\n🎮 Episode {i+1}")
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step += 1
            
            if step % 100 == 0:
                print(f"  Step {step}, Reward: {reward[0]:.3f}, Total: {total_reward:.3f}")
        
        print(f"  Episode {i+1} completed: {step} steps, Total reward: {total_reward:.3f}")
        obs = env.reset()
    
    env.close()
    print("\n🎉 Training completed!")

if __name__ == "__main__":
    main()
