# PPO vs DQN: Reinforcement Learning Comparison Project

This project compares two widely used reinforcement learning algorithms — **Proximal Policy Optimization (PPO)** and **Deep Q-Network (DQN)** — across 8 OpenAI Gym environments.

## 🎮 Environments Tested
- **Acrobot**: Swing up an underactuated pendulum
- **Blackjack**: Card game strategy optimization  
- **CartPole**: Balance a pole on a moving cart
- **CliffWalking**: Navigate safely across a cliff
- **FrozenLake**: Reach the goal on slippery ice
- **LunarLander**: Land a spacecraft safely
- **MountainCar**: Drive up a steep hill with momentum
- **Taxi**: Pick up and drop off passengers

## 🎯 Project Goals
- Train PPO and DQN agents on identical environments under controlled conditions
- Record comprehensive training metrics (rewards, success rates, training time)
- Generate detailed performance visualizations and statistical comparisons
- Produce an academic-style comparative analysis report

## 📁 Project Structure
```
project/
├── acrobot/
│   ├── acrobatPPOTrain.py
│   ├── acrobatDQNTrain.py
│   ├── ppo_model.zip
│   ├── dqn_model.zip
│   └── results/
│       ├── reward_logs.txt
│       ├── metrics.csv
│       └── reward_curves.png
├── [similar structure for other 7 environments]
├── outputs/
│   ├── avg_reward_comparison.png
│   ├── success_rate_comparison.png
│   ├── train_time_comparison.png
│   ├── std_reward_comparison.png
│   ├── final_summary_table.csv
│   └── PPO_ve_DQN_Algoritmaları_Karşılaştırma_Tablosu.pdf
├── allg.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/SedatAtakanYildiz/Reinforcement-Learning-Comparison
cd Reinforcement-Learning-Comparison
pip install -r requirements.txt
```

### Training Models
```bash
# Train PPO agent on Acrobot environment
python acrobot/acrobatPPOTrain.py

# Train DQN agent on Acrobot environment  
python acrobot/acrobatDQNTrain.py

# Generate comparison plots and summary
python allg.py
```


## 📊 Results & Analysis

The project generates comprehensive comparison outputs:

- **Performance Metrics**: Average rewards, success rates, training stability
- **Visual Comparisons**: Side-by-side performance charts across all environments
- **Statistical Analysis**: Standard deviation, convergence rates, sample efficiency
- **Summary Report**: Detailed PDF report with academic-style analysis

Key findings and insights are documented in the generated report.

## 🛠️ Technologies Used

- **Python 3.8+**
- **Stable-Baselines3**: RL algorithm implementations
- **OpenAI Gym**: Environment suite
- **PyTorch**: Neural network backend
- **Matplotlib**: Visualization and plotting
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

## 📈 Performance Metrics

The comparison includes:
- Training convergence speed
- Final performance scores
- Learning stability (reward variance)
- Sample efficiency
- Environment-specific strengths/weaknesses

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new environments
- Implement additional RL algorithms
- Improve visualization or analysis
- Report bugs or suggest features

## 📧 Contact

**Sedat Atakan Yıldız**
- Email: s.atakanyildiz@gmail.com
- GitHub: [@SedatAtakanYildiz](https://github.com/SedatAtakanYildiz)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Gym for providing the testing environments
- Stable-Baselines3 team for the RL implementations
- Research papers that inspired this comparative study

---

*This project was developed as part of reinforcement learning research to provide empirical comparisons between popular RL algorithms.*
