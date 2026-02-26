import csv
import matplotlib.pyplot as plt

def main():
    steps, rewards = [], []
    with open("outputs/logs/reward_log.csv", "r") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            rewards.append(float(row["avg_reward"]))

    plt.figure()
    plt.plot(steps, rewards)
    plt.xlabel("step")
    plt.ylabel("avg_reward")
    plt.title("RLKGF PPO: Avg Reward")
    plt.grid(True)
    plt.savefig("outputs/logs/reward_curve.png", dpi=200)
    print("Saved plot to outputs/logs/reward_curve.png")

if __name__ == "__main__":
    main()