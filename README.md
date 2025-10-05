# 🚗 Self-Driving Car with PPO Reinforcement Learning

A Pygame-based self-driving car simulation using Proximal Policy Optimization (PPO) for autonomous navigation. The car learns to drive on various tracks using deep reinforcement learning.

## ✨ Features

- **Reinforcement Learning**: Uses PPO algorithm for training
- **Multiple Track Support**: Works with different track layouts including underpasses
- **Enhanced Observations**: 17-dimensional observation space with progress tracking
- **Continuous & Discrete Actions**: Supports both action spaces
- **Headless Training**: Train models without GUI for faster training
- **Visual Demo**: Watch trained models drive in real-time
- **Domain Randomization**: Improves model robustness
- **Corner Handling**: Specialized rewards for better corner navigation

## 🎮 Controls (Manual Mode)

- **W/↑**: Accelerate forward
- **S/↓**: Reverse
- **A/←**: Turn left
- **D/→**: Turn right
- **Space**: Brake
- **ESC**: Exit

## Demo

![self-driving-car-demo](https://github.com/user-attachments/assets/06707b1c-e5dd-49f3-8427-364662043ca8)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Self-Driving-Car

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Manual Testing

```bash
# Test the environment manually
python -m env.main
```

### 3. Training a Model

```bash
# Train with continuous actions (recommended)
python train_ppo.py --continuous

# Train with discrete actions
python train_ppo.py

# Train with enhanced observations
python train_ppo.py --continuous --enhanced
```

### 4. Testing Trained Models

```bash
# Demo a trained model
python demo_model.py --model models/ppo_self_driving_continuous --episodes 3 --continuous

# Evaluate model performance
python evaluate_model.py --model models/ppo_self_driving_continuous

# Test model headlessly
python test_model.py --model models/ppo_self_driving_continuous --episodes 10
```

## 📁 Project Structure

```
Self-Drivin
├── env/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # Interactive simulation
│   ├── car.py               # Advanced car physics and controls
│   ├── track.py             # Track rendering and collision detection
│   └── gym_env.py           # Gymnasium RL environment
├── images/
│   ├── car.png              # Car sprite
│   ├── track.png            # Main track
│   ├── track2.jpg           # Alternative track
│   └── track3.jpg           # Alternative track
├── models/                  # Saved RL models
├── train_ppo.py             # PPO training script
├── basic_car_env.py         # Legacy simple simulation
└── README.md
```

## 🧠 Reinforcement Learning Details

### Environment Specification
- **Action Space**: Discrete(7) - 7 different driving actions
- **Observation Space**: Box(9) - 5 ray distances + speed + angle + position
- **Reward Function**: 
  - Living reward: +0.05 per step
  - Speed reward: +0.01 × (speed/max_speed)
  - Off-track penalty: -1.0

### Actions
0. **Nothing** - Coast
1. **Accelerate** - Forward acceleration
2. **Brake** - Deceleration
3. **Steer Left** - Turn left
4. **Steer Right** - Turn right
5. **Accelerate + Left** - Forward + turn left
6. **Accelerate + Right** - Forward + turn right

### Observations
- **Ray Sensors**: 5 normalized distances [0,1] in directions [-45°, -22.5°, 0°, 22.5°, 45°]
- **Speed**: Normalized current speed [0,1]
- **Angle**: Sine and cosine of car angle [-1,1]
- **Position**: Normalized x,y coordinates [0,1]

## ⚙️ Physics Parameters

### Tuning Options (in `env/car.py`)
```python
# Basic physics
self.max_speed = 6                    # Maximum forward speed
self.acceleration = 0.25              # Acceleration rate
self.friction = 0.05                  # Rolling friction
self.rotation_base = 3                # Base steering strength

# Enhanced handling
self.max_angular_velocity = 6.0       # Maximum turn rate
self.steer_smooth_factor = 0.25       # Steering responsiveness (0-1)
self.accel_speed_curve_coeff = 0.6    # Speed-based acceleration reduction
self.downforce_coeff = 0.12           # Speed-based grip increase

# Drift mechanics
self.handbrake_friction = 0.18        # Friction when handbraking
self.handbrake_slip = 0.45            # Slip when handbraking
```

## 🎯 Training Tips

### PPO Hyperparameters
The default configuration is optimized for this environment:
- **Learning Rate**: 3e-4
- **Batch Size**: 256
- **Steps per Update**: 1024
- **Gamma**: 0.995
- **GAE Lambda**: 0.95

### Training Progress
- **Total Timesteps**: 200,000 (adjustable)
- **Evaluation**: Automatic rendering after training
- **Model Saving**: Saves to `models/ppo_self_driving.zip`

### Customization
Modify `train_ppo.py` to:
- Change training duration
- Adjust hyperparameters
- Add custom reward functions
- Implement curriculum learning

## 🔧 Advanced Usage

### Custom Track
Replace `images/track.png` with your own track. The system uses color-based detection:
- **Road**: Non-green areas
- **Grass/Off-track**: Green-dominant areas

### Custom Car Physics
Modify `env/car.py` to adjust:
- Acceleration curves
- Steering response
- Grip characteristics
- Drift behavior

### Custom RL Environment
Extend `env/gym_env.py` to add:
- Different reward functions
- Additional observations
- Custom action spaces
- Multi-agent support

## 🐛 Troubleshooting

### Common Issues
1. **Gamepad not detected**: Ensure controller is connected before starting
2. **Track not loading**: Check image path and file existence
3. **Training slow**: Reduce `render_mode` to `None` during training
4. **Poor RL performance**: Adjust reward function or increase training time

### Dependencies
- Python 3.8+
- Pygame 2.0+
- Gymnasium 0.28+
- Stable-Baselines3 2.0+
- NumPy 1.20+

## 📈 Performance Metrics

### Training Benchmarks
- **Convergence**: ~100k timesteps for basic driving
- **Success Rate**: ~80% episodes without off-track
- **Average Episode Length**: ~500-1000 steps

### Optimization Tips
- Use vectorized environments for faster training
- Implement reward shaping for better convergence
- Add curriculum learning for complex tracks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Pygame for the game engine
- Gymnasium for RL environment standards
- Stable-Baselines3 for PPO implementation
- The reinforcement learning community for algorithms and techniques

---

**Happy Driving!**

# run env 
python -m env.main

# Train a model
python train_ppo.py --continuous --timesteps 200000

# Quick visual demo
# For continuous models
python demo_model.py --model models/ppo_self_driving_continuous_enhanced --episodes 10 --continuous

# For discrete models  
python demo_model.py --model models/ppo_self_driving_discrete --episodes 3 --discrete

# Statistical testing
python test_model.py --model models/ppo_self_driving_continuous --episodes 20 --mode headless

# Comprehensive evaluation
python evaluate_model.py --model models/ppo_self_driving_continuous --episodes 50

# Compare multiple models
python test_model.py --mode compare --models models/ppo_self_driving_continuous models/ppo_self_driving_discrete


