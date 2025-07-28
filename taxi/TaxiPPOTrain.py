import gymnasium as gym
import time
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


# Ortamı oluştur ve vektörleştir
env = DummyVecEnv([lambda: gym.make("Taxi-v3")])

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
model.save("ppo_taxi_model")

# Test ortamı
test_env = gym.make("Taxi-v3")
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

# İstatistikler
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
max_reward = np.max(rewards)
min_reward = np.min(rewards)
success_rate = np.sum(np.array(rewards) >= 7) / episodes  # 7 veya üstü ödül başarı sayılabilir

# Yazdır
print(f"Eğitim süresi: {training_time:.2f} saniye")
print(f"Ortalama Ödül: {mean_reward:.2f}")
print(f"Standart Sapma: {std_reward:.2f}")
print(f"En İyi Ödül: {max_reward}")
print(f"En Kötü Ödül: {min_reward}")
print(f"Başari Orani (>=7): {success_rate * 100:.2f}%")

# Kayıt klasörü
os.makedirs("results", exist_ok=True)

# CSV kaydı
with open("results/ppo_taxi_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["env_id", "mean_reward", "std_reward", "training_time", "success_rate", "max_reward", "min_reward"])
    writer.writerow(["Taxi-v3", mean_reward, std_reward, training_time, success_rate, max_reward, min_reward])

# Grafik
plt.plot(rewards)
plt.title("PPO - Taxi Learning Curve (500 Episode)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.savefig("results/ppo_taxi_learning_curve.png")
plt.close()