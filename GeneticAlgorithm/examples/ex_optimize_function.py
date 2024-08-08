import matplotlib.pyplot as plt
from GA import *


def my_optimization_function(
    chromosome: np.ndarray, custom_args: list = None
) -> np.float32:
    """
    The fitness function to be optimized.

    You need to know perfectly the problem you are trying to solve
    and implement the fitness function accordingly.
    """

    # How will you use the chromosome to solve the problem?
    x = chromosome[0]
    # What do you want to optimize? Minimize or maximize?
    return 1 * (x**2 - 6 * (x + 10) * np.sin(x))


ga = GA(
    generations=500,
    population_size=150,
    chromosome_length=1,
    chromosome_min_value=-1,
    chromosome_max_value=1,
    chromosome_dtype=np.float32,
    selection_method="best",
    mutation_method="arithmetical",
    optimization_type="minimize",
)

ga.evolve(my_optimization_function)
ga.print_results()
ga.plot_fitness(plt)
