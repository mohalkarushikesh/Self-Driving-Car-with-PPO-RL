#!/usr/bin/env python3
"""
Comprehensive evaluation script for self-driving car models.
Includes detailed metrics, performance analysis, and visualization.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from env.gym_env import SelfDrivingEnv
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self, model_path, continuous=True):
        self.model_path = model_path
        self.continuous = continuous
        self.model = PPO.load(model_path)
        self.results = {}
        
    def evaluate_comprehensive(self, num_episodes=50, track_variations=None):
        """
        Comprehensive evaluation with multiple metrics and track variations.
        """
        print(f"🔍 Comprehensive evaluation: {self.model_path}")
        
        if track_variations is None:
            track_variations = ["track.png", "track2.jpg", "track3.jpg"]
        
        all_results = {}
        
        for track in track_variations:
            print(f"\n🏁 Testing on track: {track}")
            results = self._evaluate_single_track(track, num_episodes)
            all_results[track] = results
        
        # Calculate overall statistics
        self.results = self._calculate_overall_stats(all_results)
        
        # Generate reports
        self._generate_text_report()
        self._generate_visual_report()
        self._save_detailed_results()
        
        return self.results
    
    def _evaluate_single_track(self, track_path, num_episodes):
        """Evaluate model on a single track."""
        env = SelfDrivingEnv(
            render_mode=None,
            continuous=self.continuous,
            visualize_sensors=False,
            domain_randomization=False,
            track_path=track_path
        )
        
        episode_data = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_info = {
                'episode': episode,
                'rewards': [],
                'actions': [],
                'observations': [],
                'positions': [],
                'speeds': [],
                'angles': [],
                'ray_distances': [],
                'total_reward': 0,
                'steps': 0,
                'completed': False,
                'collision_step': None
            }
            
            step = 0
            max_steps = 2000
            
            while step < max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Record step data
                episode_info['rewards'].append(reward)
                episode_info['actions'].append(action)
                episode_info['observations'].append(obs.copy())
                episode_info['positions'].append((info['x'], info['y']))
                episode_info['speeds'].append(info['speed'])
                episode_info['angles'].append(np.arctan2(obs[7], obs[8]))  # Extract angle from sin/cos
                episode_info['ray_distances'].append(obs[:5])  # First 5 elements are ray distances
                
                episode_info['total_reward'] += reward
                episode_info['steps'] = step + 1
                
                if terminated:
                    episode_info['completed'] = False
                    episode_info['collision_step'] = step
                    break
                elif truncated:
                    episode_info['completed'] = True
                    break
                
                step += 1
            
            episode_data.append(episode_info)
        
        env.close()
        return episode_data
    
    def _calculate_overall_stats(self, all_results):
        """Calculate overall statistics across all tracks."""
        overall_stats = {
            'tracks': {},
            'overall': {}
        }
        
        all_rewards = []
        all_lengths = []
        all_completion_rates = []
        all_speeds = []
        all_ray_distances = []
        
        for track_name, track_results in all_results.items():
            track_rewards = [ep['total_reward'] for ep in track_results]
            track_lengths = [ep['steps'] for ep in track_results]
            track_completion = [ep['completed'] for ep in track_results]
            track_speeds = [np.mean(ep['speeds']) for ep in track_results]
            track_rays = [np.mean(ep['ray_distances'], axis=0) for ep in track_results]
            
            overall_stats['tracks'][track_name] = {
                'avg_reward': np.mean(track_rewards),
                'std_reward': np.std(track_rewards),
                'avg_length': np.mean(track_lengths),
                'completion_rate': np.mean(track_completion),
                'avg_speed': np.mean(track_speeds),
                'avg_ray_distances': np.mean(track_rays, axis=0).tolist()
            }
            
            all_rewards.extend(track_rewards)
            all_lengths.extend(track_lengths)
            all_completion_rates.extend(track_completion)
            all_speeds.extend(track_speeds)
            all_ray_distances.extend(track_rays)
        
        overall_stats['overall'] = {
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'avg_length': np.mean(all_lengths),
            'completion_rate': np.mean(all_completion_rates),
            'avg_speed': np.mean(all_speeds),
            'avg_ray_distances': np.mean(all_ray_distances, axis=0).tolist()
        }
        
        return overall_stats
    
    def _generate_text_report(self):
        """Generate detailed text report."""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        print(f"Model: {self.model_path}")
        print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Action Space: {'Continuous' if self.continuous else 'Discrete'}")
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        overall = self.results['overall']
        print(f"  Average Reward: {overall['avg_reward']:.3f} ± {overall['std_reward']:.3f}")
        print(f"  Average Episode Length: {overall['avg_length']:.1f} steps")
        print(f"  Completion Rate: {overall['completion_rate']:.1%}")
        print(f"  Average Speed: {overall['avg_speed']:.2f}")
        
        print(f"\n🏁 TRACK-SPECIFIC PERFORMANCE:")
        for track_name, track_stats in self.results['tracks'].items():
            print(f"  {track_name}:")
            print(f"    Reward: {track_stats['avg_reward']:.3f} ± {track_stats['std_reward']:.3f}")
            print(f"    Length: {track_stats['avg_length']:.1f} steps")
            print(f"    Completion: {track_stats['completion_rate']:.1%}")
            print(f"    Speed: {track_stats['avg_speed']:.2f}")
        
        print(f"\n📡 SENSOR ANALYSIS:")
        ray_dists = overall['avg_ray_distances']
        angles = [-45, -22.5, 0, 22.5, 45]
        for i, (angle, dist) in enumerate(zip(angles, ray_dists)):
            print(f"  Ray {angle:>6.1f}°: {dist:.3f}")
    
    def _generate_visual_report(self):
        """Generate comprehensive visual report."""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Performance comparison across tracks
        ax1 = plt.subplot(3, 4, 1)
        tracks = list(self.results['tracks'].keys())
        rewards = [self.results['tracks'][t]['avg_reward'] for t in tracks]
        errors = [self.results['tracks'][t]['std_reward'] for t in tracks]
        ax1.bar(tracks, rewards, yerr=errors, capsize=5)
        ax1.set_title('Average Rewards by Track')
        ax1.set_ylabel('Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Completion rates
        ax2 = plt.subplot(3, 4, 2)
        completion_rates = [self.results['tracks'][t]['completion_rate'] for t in tracks]
        ax2.bar(tracks, completion_rates)
        ax2.set_title('Completion Rates by Track')
        ax2.set_ylabel('Completion Rate')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Episode lengths
        ax3 = plt.subplot(3, 4, 3)
        lengths = [self.results['tracks'][t]['avg_length'] for t in tracks]
        ax3.bar(tracks, lengths)
        ax3.set_title('Average Episode Lengths')
        ax3.set_ylabel('Steps')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Average speeds
        ax4 = plt.subplot(3, 4, 4)
        speeds = [self.results['tracks'][t]['avg_speed'] for t in tracks]
        ax4.bar(tracks, speeds)
        ax4.set_title('Average Speeds')
        ax4.set_ylabel('Speed')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Sensor ray distances
        ax5 = plt.subplot(3, 4, 5)
        angles = [-45, -22.5, 0, 22.5, 45]
        ray_dists = self.results['overall']['avg_ray_distances']
        ax5.bar(range(len(angles)), ray_dists)
        ax5.set_title('Average Ray Distances')
        ax5.set_ylabel('Distance')
        ax5.set_xticks(range(len(angles)))
        ax5.set_xticklabels([f'{a}°' for a in angles])
        
        # 6. Performance distribution
        ax6 = plt.subplot(3, 4, 6)
        all_rewards = []
        for track_results in self.results['tracks'].values():
            all_rewards.extend([ep['total_reward'] for ep in track_results])
        ax6.hist(all_rewards, bins=20, alpha=0.7, edgecolor='black')
        ax6.set_title('Reward Distribution')
        ax6.set_xlabel('Total Reward')
        ax6.set_ylabel('Frequency')
        
        # 7-12. Additional analysis plots (placeholder for future metrics)
        for i in range(7, 13):
            ax = plt.subplot(3, 4, i)
            ax.text(0.5, 0.5, f'Analysis {i-6}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Additional Metric {i-6}')
        
        plt.tight_layout()
        plt.savefig('evaluation_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visual report saved as 'evaluation_report.png'")
    
    def _save_detailed_results(self):
        """Save detailed results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'evaluation_results_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"💾 Detailed results saved to '{filename}'")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes per track')
    parser.add_argument('--continuous', action='store_true', help='Continuous action space')
    parser.add_argument('--tracks', nargs='+', help='Specific tracks to test')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model + ".zip"):
        print(f"❌ Model file not found: {args.model}.zip")
        return
    
    evaluator = ModelEvaluator(args.model, args.continuous)
    results = evaluator.evaluate_comprehensive(args.episodes, args.tracks)
    
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    main() 
