import numpy as np
import matplotlib.pyplot as plt
from PSO import PSO

np.random.seed(42)


def my_optimization_function(chromosome: np.ndarray) -> np.float32:
    """
    The fitness function to be optimized.

    You need to know perfectly the problem you are trying to solve
    and implement the fitness function accordingly.
    """

    # How will you use the chromosome to solve the problem?
    x = chromosome[0]
    # What do you want to optimize? Minimize or maximize?
    return x**2 - 6 * (x + 10) * np.sin(x)


# Initialize the genetic algorithm
pso = PSO(
    paticle_size=15,
    particle_dimension=1,
    min_value=-20,
    max_value=0,
    fitness_function=my_optimization_function,
    self_confidence_c1=1.2,
    swarm_confidence_c2=1.2,
    inertia_weight_w=0.5,
    iterations=50,
    minimize=True,
    limit_search_space=False,
)

best_fitness, best_position = pso.optimize()
print("Best position: ", best_position[-1])
print("Best fitness: ", best_fitness[-1])
pso.plot_fitness()
