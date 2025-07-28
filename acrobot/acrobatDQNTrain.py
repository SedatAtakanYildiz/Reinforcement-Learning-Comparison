import gymnasium as gym
import time
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import torch


# Ortamı oluştur
env = gym.make("Acrobot-v1")

# DQN modelini oluştur
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0004,
    gamma=0.99,
    buffer_size=100_000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    batch_size=64,
    train_freq=1,
    target_update_interval=500,
    verbose=1,
    device="cpu"
)

# Eğitimi başlat
start_time = time.time()
model.learn(total_timesteps=1_000_000)
training_time = time.time() - start_time
model.save("dqn_acrobot_model")

# Test ortamı
test_env = gym.make("Acrobot-v1")
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
success_rate = np.sum(np.array(rewards) >= -100) / episodes  # örnek başarı eşiği

# Yazdır
print(f"Eğitim süresi: {training_time:.2f} saniye")
print(f"Ortalama Ödül: {mean_reward:.2f}")
print(f"Standart Sapma: {std_reward:.2f}")
print(f"En İyi Ödül: {max_reward}")
print(f"En Kötü Ödül: {min_reward}")
print(f"Başarı Oranı (>= -100): {success_rate * 100:.2f}%")

# Kayıt klasörü
os.makedirs("results", exist_ok=True)

# CSV kaydı
with open("results/dqn_acrobot_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["env_id", "mean_reward", "std_reward", "training_time", "success_rate", "max_reward", "min_reward"])
    writer.writerow(["Acrobot-v1", mean_reward, std_reward, training_time, success_rate, max_reward, min_reward])

# Grafik
plt.plot(rewards)
plt.title("DQN - Acrobot Learning Curve (500 Episode)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.savefig("results/dqn_acrobot_learning_curve.png")
plt.close()