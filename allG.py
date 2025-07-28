# Yeniden başlatma sonrası gerekli kütüphaneleri ve verileri tekrar yükleyip çalıştırıyoruz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Verileri tekrar tanımlayalım
data = [
    ["Acrobot-v1", "PPO", -65.62, 9.63, 1130.92, 99.20, -61, -178],
    ["Acrobot-v1", "DQN", -88.22, 15.08, 2294.83, 81.00, -68, -166],
    ["Blackjack-v1", "PPO", -0.0560, 0.9428, 1067.57, 41.80, 1.0, -1.0],
    ["Blackjack-v1", "DQN", -0.0100, 0.9455, 2340.66, 44.20, 1.0, -1.0],
    ["CartPole-v1", "PPO", 500.00, 0.0, 900.33, 100.00, 500, 500],
    ["CartPole-v1", "DQN", 148.23, 7.10, 1998.01, 0.00, 160, 23],
    ["CliffWalking-v0", "PPO", -13.00, 0.0, 968.94, 100.00, -13, -13],
    ["CliffWalking-v0", "DQN", -13.00, 0.0, 2217.13, 100.00, -13, -13],
    ["FrozenLake-v1", "PPO", 0.7340, 0.4419, 861.97, 73.40, 1.0, 0.0],
    ["FrozenLake-v1", "DQN", 0.7560, 0.4295, 2149.35, 75.60, 1.0, 0.0],
    ["LunarLander-v3", "PPO", 240.31, 72.13, 1007.19, 87.40, 322.67, -57.71],
    ["LunarLander-v3", "DQN", 239.01, 66.20, 2128.51, 78.60, 316.70, 4.29],
    ["MountainCar-v0", "PPO", -101.66, 10.25, 1062.47, 84.20, -83, -114],
    ["MountainCar-v0", "DQN", -200.00, 0.0, 2215.81, 0.00, -200, -200],
    ["Taxi-v3", "PPO", -200.00, 0.0, 1729.55, 0.00, -200, -200],
    ["Taxi-v3", "DQN", 7.84, 2.49, 2753.76, 69.00, 15, 3],
]

df = pd.DataFrame(data, columns=[
    "Environment", "Algorithm", "MeanReward", "StdReward", "TrainTime",
    "SuccessRate", "MaxReward", "MinReward"
])

# Kayıt klasörü
os.makedirs("outputs", exist_ok=True)

# Ortalama ödül grafiği
plt.figure(figsize=(12, 6))
for alg in ["PPO", "DQN"]:
    subset = df[df["Algorithm"] == alg]
    plt.bar(
        [f"{env}\n({alg})" for env in subset["Environment"]],
        subset["MeanReward"],
        label=alg,
        alpha=0.7
    )
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average Reward")
plt.title("Average Reward by Environment and Algorithm")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/avg_reward_comparison.png")
plt.close()

# Eğitim süresi grafiği
plt.figure(figsize=(12, 6))
for alg in ["PPO", "DQN"]:
    subset = df[df["Algorithm"] == alg]
    plt.bar(
        [f"{env}\n({alg})" for env in subset["Environment"]],
        subset["TrainTime"],
        label=alg,
        alpha=0.7
    )
plt.xticks(rotation=45, ha="right")
plt.ylabel("Training Time (sec)")
plt.title("Training Time by Environment and Algorithm")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/train_time_comparison.png")
plt.close()

# Başarı oranı grafiği
plt.figure(figsize=(12, 6))
for alg in ["PPO", "DQN"]:
    subset = df[df["Algorithm"] == alg]
    plt.bar(
        [f"{env}\n({alg})" for env in subset["Environment"]],
        subset["SuccessRate"],
        label=alg,
        alpha=0.7
    )
plt.xticks(rotation=45, ha="right")
plt.ylabel("Success Rate (%)")
plt.title("Success Rate by Environment and Algorithm")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/success_rate_comparison.png")
plt.close()

# Standart sapma grafiği
plt.figure(figsize=(12, 6))
for alg in ["PPO", "DQN"]:
    subset = df[df["Algorithm"] == alg]
    plt.bar(
        [f"{env}\n({alg})" for env in subset["Environment"]],
        subset["StdReward"],
        label=alg,
        alpha=0.7
    )
plt.xticks(rotation=45, ha="right")
plt.ylabel("Std. Reward")
plt.title("Standard Deviation of Reward by Environment and Algorithm")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/std_reward_comparison.png")
plt.close()

# Final tabloyu CSV olarak kaydet
df.to_csv("outputs/final_summary_table.csv", index=False)

"Grafikler ve özet tablo başarıyla oluşturuldu."
