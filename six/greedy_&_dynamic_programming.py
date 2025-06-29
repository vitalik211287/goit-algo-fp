def greedy_algorithm(items, budget):
    sorted_items = sorted(items.items(), key=lambda x: x[1]['calories'] / x[1]['cost'], reverse=True)
    total_cost = 0
    result = []
    for name, data in sorted_items:
        if total_cost + data['cost'] <= budget:
            result.append(name)
            total_cost += data['cost']
    return result

def dynamic_programming(items, budget):
    names = list(items.keys())
    dp = [0] * (budget + 1)
    keep = [None] * (budget + 1)
    for i, name in enumerate(names):
        cost = items[name]['cost']
        calories = items[name]['calories']
        for b in range(budget, cost - 1, -1):
            if dp[b - cost] + calories > dp[b]:
                dp[b] = dp[b - cost] + calories
                keep[b] = i

    result = []
    b = budget
    while b >= 0 and keep[b] is not None:
        i = keep[b]
        name = names[i]
        result.append(name)
        b -= items[name]['cost']
    return result[::-1]

print("\n=== Knapsack Example ===")
food_items = {
    'Burger': {'cost': 3, 'calories': 500},
    'Salad': {'cost': 2, 'calories': 150},
    'Fries': {'cost': 1, 'calories': 300},
    'Steak': {'cost': 5, 'calories': 700}
}
budget = 6
print("Greedy result:", greedy_algorithm(food_items, budget))
print("DP result:", dynamic_programming(food_items, budget))