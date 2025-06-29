import random
# import pandas as pd
import matplotlib.pyplot as plt

def monte_carlo_simulation(throws=10000):
    counts = {i: 0 for i in range(2, 13)}
    for _ in range(throws):
        dice_sum = random.randint(1, 6) + random.randint(1, 6)
        counts[dice_sum] += 1
    probabilities = {k: v / throws for k, v in counts.items()}
    return probabilities


def plot_probabilities(probabilities):
    plt.bar(probabilities.keys(), probabilities.values(), color='skyblue')
    plt.title("Monte Carlo Dice Sum Probabilities")
    plt.xlabel("Sum")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.show()

# Виклик функції після симуляції
probs = monte_carlo_simulation(5000)
plot_probabilities(probs)


print("\n=== Monte Carlo Dice ===")
probs = monte_carlo_simulation(5000)
for k in sorted(probs):
    print(f"Sum {k}: {probs[k]:.4f}")