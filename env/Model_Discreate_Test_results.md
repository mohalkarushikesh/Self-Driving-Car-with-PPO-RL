```
❌ Episode ended: Max steps reached (5000)
------------------------------
| time/              |       |
|    fps             | 98    |
|    iterations      | 47    |
|    time_elapsed    | 981   |
|    total_timesteps | 96256 |
------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 99          |
|    iterations           | 48          |
|    time_elapsed         | 987         |
|    total_timesteps      | 98304       |
| train/                  |             |
|    approx_kl            | 0.007027518 |
|    clip_fraction        | 0.0767      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.694      |
|    explained_variance   | 0.984672    |
|    learning_rate        | 0.0003      |
|    loss                 | -0.00452    |
|    n_updates            | 470         |
|    policy_gradient_loss | -0.00184    |
|    value_loss           | 0.00727     |
-----------------------------------------
❌ Episode ended: Max steps reached (5000)
❌ Episode ended: Max steps reached (5000)
❌ Episode ended: Max steps reached (5000)
❌ Episode ended: Max steps reached (5000)
❌ Episode ended: Max steps reached (5000)
Eval num_timesteps=100000, episode_reward=4476.51 +/- 0.00
Episode length: 5000.00 +/- 0.00
------------------------------------------
| eval/                   |              |
|    mean_ep_length       | 5e+03        |
|    mean_reward          | 4.48e+03     |
| time/                   |              |
|    total_timesteps      | 100000       |
| train/                  |              |
|    approx_kl            | 0.0067342427 |
|    clip_fraction        | 0.118        |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.699       |
|    explained_variance   | 0.96418065   |
|    learning_rate        | 0.0003       |
|    loss                 | -0.0202      |
|    n_updates            | 480          |
|    policy_gradient_loss | -0.000824    |
|    value_loss           | 0.00884      |
------------------------------------------
❌ Episode ended: Max steps reached (5000)
-------------------------------
| time/              |        |
|    fps             | 97     |
|    iterations      | 49     |
|    time_elapsed    | 1031   |
|    total_timesteps | 100352 |
-------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100,352/100,000  [ 0:17:38 < 0:00:00 , 135 it/s ]
💾 Model saved to: models/ppo_self_driving_discrete

🎮 Testing trained model...

🎮 Episode 1
  Step 100, Reward: 0.879, Total: 75.201
  Step 200, Reward: 0.899, Total: 163.989
  Step 300, Reward: 0.883, Total: 254.625
  Step 400, Reward: 0.882, Total: 344.289
  Step 500, Reward: 0.913, Total: 433.676
  Step 600, Reward: 0.891, Total: 523.398
  Step 700, Reward: 0.915, Total: 613.925
  Step 800, Reward: 0.927, Total: 704.747
  Step 900, Reward: 0.888, Total: 793.675
  Step 1000, Reward: 0.867, Total: 882.937
  Episode 1 completed: 1000 steps, Total reward: 882.937

🎮 Episode 2
  Step 100, Reward: 0.879, Total: 75.201
  Step 200, Reward: 0.899, Total: 163.989
  Step 300, Reward: 0.883, Total: 254.625
  Step 400, Reward: 0.882, Total: 344.289
  Step 500, Reward: 0.913, Total: 433.676
  Step 600, Reward: 0.891, Total: 523.398
  Step 700, Reward: 0.915, Total: 613.925
  Step 800, Reward: 0.927, Total: 704.747
  Step 900, Reward: 0.888, Total: 793.675
  Step 1000, Reward: 0.867, Total: 882.937
  Episode 2 completed: 1000 steps, Total reward: 882.937

🎮 Episode 3
  Step 100, Reward: 0.879, Total: 75.201
  Step 200, Reward: 0.899, Total: 163.989
  Step 300, Reward: 0.883, Total: 254.625
  Step 400, Reward: 0.882, Total: 344.289
  Step 500, Reward: 0.913, Total: 433.676
  Step 600, Reward: 0.891, Total: 523.398
  Step 700, Reward: 0.915, Total: 613.925
  Step 800, Reward: 0.927, Total: 704.747
  Step 900, Reward: 0.888, Total: 793.675
  Step 1000, Reward: 0.867, Total: 882.937
  Episode 3 completed: 1000 steps, Total reward: 882.937

🎉 Training completed!

```