import numpy as np
import matplotlib.pyplot as plt
from PSO import PSO


def plot_cities(cities):
    for city, (x, y) in cities.items():
        plt.plot(x, y, "ro")
        plt.text(x, y, city)
    plt.axis("equal")
    plt.show()


def tsp_fitness_function(chromosome: np.ndarray) -> np.float32:

    # cities = custom_args[0]
    cities = {
        "A": (0, 0),
        "B": (1, 1),
        "C": (-1, 0),
        "D": (0, 1),
        "E": (-1, 2),
        "F": (3, -1),
        "G": (-2, 1),
        "H": (1, -2),
        "I": (2, 1),
        "J": (-2, -1),
    }
    num_cities = len(cities)

    # Ensure closed tour
    if chromosome[0] != chromosome[-1]:
        return np.inf

    # Ensure not repeated cities (ignoring the last element which should be same as the first)
    if len(set(chromosome[:-1])) != num_cities:
        return np.inf

    # Ensure all cities are visited
    if len(chromosome) != num_cities + 1:
        return np.inf

    # # Time taken: 12
    # dist_matrix = custom_args[1]
    # distance = np.sum(
    #     [
    #         dist_matrix[chromosome[i], chromosome[i + 1]]
    #         for i in range(len(chromosome) - 1)
    #     ]
    # )
    # return distance

    # Time taken: 19.33
    city_coords = np.array(list(cities.values()))
    distance = np.sum(
        np.sqrt(
            np.sum(
                (city_coords[chromosome[:-1]] - city_coords[chromosome[1:]]) ** 2,
                axis=1,
            )
        )
    )
    return distance


def plot_tour(tour, cities):
    for i in range(len(tour) - 1):
        city1 = list(cities.keys())[tour[i]]
        city2 = list(cities.keys())[tour[i + 1]]
        plt.plot(
            [cities[city1][0], cities[city2][0]],
            [cities[city1][1], cities[city2][1]],
            "k-",
        )
    plot_cities(cities)


cities = {
    "A": (0, 0),
    "B": (1, 1),
    "C": (-1, 0),
    "D": (0, 1),
    "E": (-1, 2),
    "F": (3, -1),
    "G": (-2, 1),
    "H": (1, -2),
    "I": (2, 1),
    "J": (-2, -1),
}
np.random.seed(42)
pso = PSO(
    paticle_size=100,
    particle_dimension=len(cities) + 1,
    min_value=0,
    max_value=len(cities)-1,
    fitness_function=tsp_fitness_function,
    self_confidence_c1=1,
    swarm_confidence_c2=8,
    inertia_weight_w=0.135,
    iterations=3700,
    minimize=True,
    limit_search_space=True,
    dtype=np.int32,
    verbose=True,
)

best_fitness, best_position = pso.optimize()
print("\nBest position: ", best_position[-1])
print("Best fitness: ", best_fitness[-1])
# pso.plot_fitness()
best_position = best_position.astype(int)
cities_selected = [list(cities.keys())[i] for i in best_position[-1]]
print("Cities selected: ", cities_selected)
plot_tour(best_position[-1], cities)
