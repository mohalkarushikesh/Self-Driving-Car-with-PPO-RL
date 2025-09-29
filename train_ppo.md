import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env.gym_env import SelfDrivingEnv

# 👀 Optional: Render during training (disable for headless mode)
# class RenderCallback(BaseCallback):
#     def __init__(self, render_freq=1, verbose=0):
#         super().__init__(verbose)
#         self.render_freq = render_freq

#     def _on_step(self) -> bool:
#         if self.n_calls % self.render_freq == 0:
#             try:
#                 self.training_env.envs[0].render()
#             except Exception:
#                 pass
#         return True

## �� **Key Fixes Made:*

# 🧱 Environment factory
def make_env(continuous=True, visualize=False, domain_randomization=False):
    def _thunk():
        return SelfDrivingEnv(
            render_mode=None,  # Use None for headless training
            continuous=continuous,
            visualize_sensors=visualize,
            domain_randomization=domain_randomization,
            enhanced_observations=True  # Use enhanced observations for new training
        )
    return _thunk

def main():
    parser = argparse.ArgumentParser(description='Train self-driving car with PPO')
    parser.add_argument("--continuous", action="store_true", help="Use continuous action space")
    parser.add_argument("--visualize", action="store_true", help="Visualize sensors during training")
    parser.add_argument("--domain_randomization", action="store_true", help="Randomize physics per episode")
    parser.add_argument("--timesteps", type=int, default=400_000, help="Total training timesteps")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint_freq", type=int, default=50_000, help="Checkpoint frequency")
    args = parser.parse_args()

    print("🚗 Self-Driving Car Training (Enhanced)")
    print("=" * 50)
    print("✨ Improvements:")
    print("  - Smoothness penalties to reduce shaking")
    print("  - Progress-based rewards")
    print("  - Enhanced observation space (17 dimensions)")
    print("  - Better center tracking")
    print("  - Corner handling rewards")
    print("=" * 50)

    # 🧠 Create training environment
    env = DummyVecEnv([
        make_env(
            continuous=args.continuous,
            visualize=args.visualize,
            domain_randomization=args.domain_randomization
        )
    ])

    # 🧠 Load or initialize PPO model with better hyperparameters
    checkpoint_path = "models/ppo_checkpoint_100k"
    if args.resume and os.path.exists(checkpoint_path + ".zip"):
        print("✅ Resuming from checkpoint...")
        model = PPO.load(checkpoint_path, env=env)
    else:
        print("🚀 Starting fresh training...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=2048,  # Increased for better learning
            batch_size=512,  # Increased batch size
            gamma=0.99,  # Slightly higher discount
            gae_lambda=0.95,
            ent_coef=0.01,  # Small entropy bonus for exploration
            learning_rate=2e-4,  # Slightly lower learning rate
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,  # Gradient clipping
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])  # Fixed network architecture
            )
        )

    # 🏁 Train the model in chunks with checkpoints
    total_timesteps = args.timesteps
    chunk_size = args.checkpoint_freq
    chunks = total_timesteps // chunk_size
    
    print(f"\n📚 Training in {chunks} chunks of {chunk_size:,} timesteps each...")
    
    for chunk in range(chunks):
        print(f"\n🔄 Training chunk {chunk + 1}/{chunks}")
        
        # Train for one chunk
        model.learn(total_timesteps=chunk_size)
        
        # Save checkpoint
        os.makedirs("models", exist_ok=True)
        checkpoint_name = f"models/ppo_checkpoint_{chunk_size * (chunk + 1) // 1000}k"
        model.save(checkpoint_name)
        print(f"💾 Saved checkpoint: {checkpoint_name}")

    # 💾 Save final model
    final_path = "models/ppo_self_driving_continuous" if args.continuous else "models/ppo_self_driving_discrete"
    model.save(final_path)
    print(f"\n✅ Final model saved to: {final_path}")
    
    print("\n🎉 Training completed!")
    print("📝 Use 'python demo_model.py --model <path> --enhanced' to test the model")
    print("📊 Use 'python evaluate_model.py --model <path>' for detailed evaluation")

if __name__ == "__main__":
    main()
