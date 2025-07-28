import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Tuple, Box
import time
import csv
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

# Gözlemi düzleştirmek için wrapper
class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, Tuple)
        self.observation_space = Box(low=0, high=1, shape=(len(env.observation_space.spaces),), dtype=np.float32)

    def observation(self, obs):
        return np.array(obs, dtype=np.float32)


# Ortamı oluştur ve gözlemi düzleştir
env = DummyVecEnv([lambda: FlattenObservation(gym.make("Blackjack-v1", sab=True))])

# PPO modelini oluştur
model = PPO(
    policy="MlpPolicy",
    env=env,
    device="cpu",
    learning_rate=0.0004,
    gamma=0.99,
    gae_lambda=0.95,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    verbose=1
)

# Eğitimi başlat
start_time = time.time()
model.learn(total_timesteps=1_000_000)
training_time = time.time() - start_time
model.save("ppo_blackjack_model")

# Test ortamı (düzleştirilmiş)
test_env = FlattenObservation(gym.make("Blackjack-v1", sab=True))
episodes = 500
rewards = []

for _ in range(episodes):
    obs, _ = test_env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action) if isinstance(action, (np.ndarray, list)) else action
        obs, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward
    rewards.append(total_reward)

# Sonuçlar
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
max_reward = np.max(rewards)
min_reward = np.min(rewards)
success_rate = np.sum(np.array(rewards) > 0) / episodes

print(f"Eğitim süresi: {training_time:.2f} saniye")
print(f"Ortalama Ödül: {mean_reward:.4f}")
print(f"Standart Sapma: {std_reward:.4f}")
print(f"En İyi Ödül: {max_reward}")
print(f"En Kötü Ödül: {min_reward}")
print(f"Başarı Oranı (> 0): {success_rate * 100:.2f}%")

# Kayıt klasörü
os.makedirs("results", exist_ok=True)

# CSV
with open("results/ppo_blackjack_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["env_id", "mean_reward", "std_reward", "training_time", "success_rate", "max_reward", "min_reward"])
    writer.writerow(["Blackjack-v1", mean_reward, std_reward, training_time, success_rate, max_reward, min_reward])

# Grafik
plt.plot(rewards)
plt.title("PPO - Blackjack Learning Curve (500 Episode)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.savefig("results/ppo_blackjack_learning_curve.png")
plt.close()
