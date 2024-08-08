import numpy as np
import matplotlib.pyplot as plt
from PSO import PSO

np.random.seed(42)

# Define the items and capacity for the knapsack problem
items = [
    {"name": "glasses", "value": 60, "weight": 10},
    {"name": "backpack", "value": 100, "weight": 20},
    {"name": "cup", "value": 120, "weight": 30},
    {"name": "tv", "value": 50, "weight": 10},
    {"name": "cellphone", "value": 150, "weight": 40},
    {"name": "laptop", "value": 200, "weight": 50},
    {"name": "mouse", "value": 60, "weight": 10},
    # Add more items as needed
]

capacity = 110  # The capacity of the knapsack


def knapsack_results(best_chromosome: np.ndarray = None) -> None:
    selected_items = [
        item["name"] for gene, item in zip(best_chromosome, items) if gene == 1
    ]
    print("Selected items: ", selected_items)
    overall_value = sum(
        [item["value"] for gene, item in zip(best_chromosome, items) if gene == 1]
    )
    print("Value : ", overall_value, "USD")
    overall_weight = sum(
        [item["weight"] for gene, item in zip(best_chromosome, items) if gene == 1]
    )
    print("Weight: ", overall_weight, "kg")


def knapsack_fitness_function(
    chromosome: np.ndarray, custom_args: list = None
) -> np.float32:
    """
    Calculate the fitness of a chromosome for the knapsack problem.

    Parameters
    ----------
    chromosome : np.ndarray
        The chromosome to evaluate. In this case, it represents the items to select.
    custom_args : list, optional
        Custom arguments to pass to the fitness function, by default None.

    Returns
    -------
    np.float32
        The fitness value of the chromosome
    """

    total_value = 0
    total_weight = 0

    for gene, item in zip(chromosome, items):
        if gene == 1:
            total_value += item["value"]
            total_weight += item["weight"]

    if total_weight > capacity:
        return -1

    # return total_value
    # convert total value to minimization problem
    return total_value


def knapsack_chromosome() -> np.ndarray:
    """
    Create a chromosome.

    Returns
    -------
    np.ndarray
        The created chromosome.
    """
    # Para el problema de la mochila, los cromosomas son vectores binarios
    chromosome = np.random.choice([0, 1], size=len(items)).astype(np.int32)
    return chromosome


# Initialize the genetic algorithm
pso = PSO(
    paticle_size=5,
    particle_dimension=len(items),
    min_value=0,
    max_value=1,
    fitness_function=knapsack_fitness_function,
    self_confidence_c1=1.2,
    swarm_confidence_c2=1.52,
    inertia_weight_w=0.9,
    iterations=30,
    minimize=False,
    dtype=np.int8,
)

best_fitness, best_position = pso.optimize()
# print("Best position: ", best_position[-1])
# print("Best fitness: ", best_fitness[-1])
items_selected = [
    item["name"] for gene, item in zip(best_position[-1], items) if gene == 1
]
weight = sum(
    [item["weight"] for gene, item in zip(best_position[-1], items) if gene == 1]
)
print("Items selected: ", items_selected)
print("Weight: ", weight)
pso.plot_fitness()

