#!/usr/bin/env python3
"""
Test script for evaluating trained self-driving car models.
Supports both headless testing and visual evaluation.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.gym_env import SelfDrivingEnv

def test_model_headless(model_path, num_episodes=10, continuous=True):
    """
    Test model performance in headless mode for statistical analysis.
    """
    print(f"🧪 Testing model: {model_path}")
    
    # Create headless environment
    env = SelfDrivingEnv(
        render_mode=None,
        continuous=continuous,
        visualize_sensors=False,
        domain_randomization=False
    )
    
    # Load model
    model = PPO.load(model_path)
    
    episode_rewards = []
    episode_lengths = []
    completion_rates = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 2000
        
        while steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        completion_rates.append(1.0 if not terminated else 0.0)
        
        print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}, Completed={not terminated}")
    
    env.close()
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    completion_rate = np.mean(completion_rates)
    
    print(f"\n📊 Test Results ({num_episodes} episodes):")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.1f} steps")
    print(f"Completion Rate: {completion_rate:.1%}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'completion_rates': completion_rates,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'completion_rate': completion_rate
    }

def test_model_visual(model_path, continuous=True, visualize_sensors=True, num_episodes=3):
    """
    Test model with visual rendering for qualitative evaluation.
    """
    print(f"🎬 Visual testing: {model_path}")
    
    # Create visual environment
    env = SelfDrivingEnv(
        render_mode="human",
        continuous=continuous,
        visualize_sensors=visualize_sensors,
        domain_randomization=False
    )
    
    # Load model
    model = PPO.load(model_path)
    
    for episode in range(num_episodes):
        print(f"\n🎮 Episode {episode + 1}")
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 2000
        
        while steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Render the environment
            env.render()
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} completed: Reward={total_reward:.2f}, Steps={steps}")
        
        # Wait for user input between episodes
        input("Press Enter to continue to next episode...")
    
    env.close()

def compare_models(model_paths, num_episodes=10):
    """
    Compare multiple models and generate performance plots.
    """
    print("📈 Comparing multiple models...")
    
    results = {}
    
    for model_path in model_paths:
        if os.path.exists(model_path + ".zip"):
            model_name = os.path.basename(model_path)
            print(f"\nTesting {model_name}...")
            results[model_name] = test_model_headless(model_path, num_episodes)
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    # Generate comparison plots
    if len(results) > 1:
        plot_model_comparison(results)

def plot_model_comparison(results):
    """
    Generate comparison plots for multiple models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    model_names = list(results.keys())
    
    # Reward comparison
    rewards = [results[name]['episode_rewards'] for name in model_names]
    axes[0, 0].boxplot(rewards, labels=model_names)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Episode length comparison
    lengths = [results[name]['episode_lengths'] for name in model_names]
    axes[0, 1].boxplot(lengths, labels=model_names)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Completion rate comparison
    completion_rates = [results[name]['completion_rate'] for name in model_names]
    axes[1, 0].bar(model_names, completion_rates)
    axes[1, 0].set_title('Completion Rates')
    axes[1, 0].set_ylabel('Completion Rate')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Average reward comparison
    avg_rewards = [results[name]['avg_reward'] for name in model_names]
    std_rewards = [results[name]['std_reward'] for name in model_names]
    axes[1, 1].bar(model_names, avg_rewards, yerr=std_rewards, capsize=5)
    axes[1, 1].set_title('Average Rewards')
    axes[1, 1].set_ylabel('Average Reward')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Comparison plots saved as 'model_comparison.png'")

def main():
    parser = argparse.ArgumentParser(description='Test self-driving car models')
    parser.add_argument('--model', type=str, required=True, help='Path to model file (without .zip extension)')
    parser.add_argument('--mode', choices=['headless', 'visual', 'compare'], default='headless',
                       help='Testing mode')
    parser.add_argument('--episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--continuous', action='store_true', help='Use continuous action space')
    parser.add_argument('--visualize', action='store_true', help='Visualize sensors during visual testing')
    parser.add_argument('--models', nargs='+', help='Multiple model paths for comparison')
    
    args = parser.parse_args()
    
    if args.mode == 'headless':
        test_model_headless(args.model, args.episodes, args.continuous)
    elif args.mode == 'visual':
        test_model_visual(args.model, args.continuous, args.visualize, args.episodes)
    elif args.mode == 'compare':
        if args.models:
            compare_models(args.models, args.episodes)
        else:
            print("❌ Please provide multiple model paths with --models for comparison")

if __name__ == "__main__":
    main() 
