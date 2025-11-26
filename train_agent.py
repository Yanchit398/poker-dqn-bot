import os
import csv
import rlcard
import torch
from rlcard.agents import RandomAgent
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.utils import set_seed, reorganize

# Try to import rule-based Limit Hold'em agent
try:
    from rlcard.agents import LimitholdemRuleAgent
    RULE_AGENT_AVAILABLE = True
except Exception:
    RULE_AGENT_AVAILABLE = False


def train(num_episodes=3000):
    set_seed(42)

    # Environment for training
    env = rlcard.make(
        "limit-holdem",
        config={
            "seed": 42,
        },
    )

    # Model 1: DQN agent and opponent (Random)
    dqn_agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
    )
    random_agent = RandomAgent(num_actions=env.num_actions)

    # Player 0 = DQN, Player 1 = Random
    env.set_agents([dqn_agent, random_agent])

    train_history = []  # (episode, payoff)

    # -------- TRAIN DQN --------
    for episode in range(num_episodes):
        trajectories, payoffs = env.run(is_training=True)

        # convert trajectories format
        trajectories = reorganize(trajectories, payoffs)

        for ts in trajectories[0]:
            dqn_agent.feed(ts)

        train_history.append((episode + 1, payoffs[0]))

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, payoff = {payoffs[0]}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "dqn_limit_holdem.pth")
    torch.save(dqn_agent, model_path)
    print("✅ Training finished. Model saved to:", model_path)

    # Save training history
    history_path = os.path.join("models", "training_history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "payoff"])
        for ep, payoff in train_history:
            writer.writerow([ep, payoff])
    print("✅ Training history saved to:", history_path)

    # Evaluate two models and save comparison
    evaluate_models(dqn_agent)


def evaluate_models(dqn_agent, num_eval_episodes=200):
    """
    Compare:
      - Model 1: DQN (trained)
      - Model 2: Baseline (rule-based or random)
    """
    print("▶ Starting evaluation of two models...")

    # env for evaluation
    env_eval = rlcard.make(
        "limit-holdem",
        config={
            "seed": 123,
        },
    )
    random_opponent = RandomAgent(num_actions=env_eval.num_actions)

    # baseline model
    if RULE_AGENT_AVAILABLE:
        baseline_agent = LimitholdemRuleAgent(num_actions=env_eval.num_actions)
        baseline_name = "Rule-based"
    else:
        baseline_agent = RandomAgent(num_actions=env_eval.num_actions)
        baseline_name = "Random"

    results = []

    # ---- MODEL 1: DQN vs Random ----
    env_eval.set_agents([dqn_agent, random_opponent])
    total = 0.0
    for i in range(num_eval_episodes):
        _, payoffs = env_eval.run(is_training=False)
        total += payoffs[0]
        avg = total / (i + 1)
        results.append({"episode": i + 1, "model": "DQN", "avg_payoff": avg})

    # ---- MODEL 2: Baseline vs Random ----
    env_eval = rlcard.make(
        "limit-holdem",
        config={
            "seed": 456,
        },
    )
    env_eval.set_agents([baseline_agent, random_opponent])
    total = 0.0
    for i in range(num_eval_episodes):
        _, payoffs = env_eval.run(is_training=False)
        total += payoffs[0]
        avg = total / (i + 1)
        results.append({"episode": i + 1, "model": baseline_name, "avg_payoff": avg})

    comp_path = os.path.join("models", "comparison_history.csv")
    with open(comp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "model", "avg_payoff"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("✅ Comparison history saved to:", comp_path)
    print("   Models compared: DQN vs", baseline_name)


if __name__ == "__main__":
    train(num_episodes=3000)
