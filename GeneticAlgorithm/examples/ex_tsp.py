import matplotlib.pyplot as plt
from genetic_algorithm import *


def tsp_results(best_chromosome: np.ndarray = None) -> None:
    my_tour = [list(cities.keys())[i] for i in ga.best_chromosome]
    print(my_tour)
    print(ga.best_fitness)
    plot_tour(ga.best_chromosome, cities)

def set_cities():
    cities = {
        "A": (0, 0),
        "B": (1, 1),
        "C": (-1, 0),
        "D": (0, 1),
        "E": (-1, 2),
        # "F": (3, -1),
        # "G": (-2, 1),
        # "H": (1, -2),
        # "I": (2, 1),
        # "J": (-2, -1),
    }
    return cities


def plot_cities(cities):
    for city, (x, y) in cities.items():
        plt.plot(x, y, "ko")
        plt.text(x, y, city)
    plt.axis("equal")
    plt.show()


def tsp_chromosome() -> np.ndarray:
    """
    Create a chromosome.

    Returns
    -------
    np.ndarray
        The created chromosome.
    """

    cities = set_cities()
    tour = np.random.permutation(len(cities))
    tour = np.append(tour, tour[0])
    return tour


def tsp_fitness_function(
    chromosome: np.ndarray, custom_args: list = None
) -> np.float32:

    cities = custom_args[0]
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

    # # Time taken: 25.97
    # distance = 0
    # for i in range(len(chromosome) - 1):
    #     city1 = list(cities.keys())[chromosome[i]]
    #     city2 = list(cities.keys())[chromosome[i + 1]]
    #     distance += np.sqrt(
    #         (cities[city1][0] - cities[city2][0]) ** 2
    #         + (cities[city1][1] - cities[city2][1]) ** 2
    #     )
    # return distance

    # Time taken: 12
    dist_matrix = custom_args[1]
    distance = np.sum(
        [
            dist_matrix[chromosome[i], chromosome[i + 1]]
            for i in range(len(chromosome) - 1)
        ]
    )
    return distance

    # # Time taken: 19.33
    # city_coords = np.array(list(cities.values()))
    # distance = np.sum(np.sqrt(
    #     np.sum((city_coords[chromosome[:-1]] - city_coords[chromosome[1:]])**2, axis=1)
    # ))
    # return distance


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


ga = GeneticAlgorithm(
    generations=50,
    population_size=15000,
    chromosome_length=len(set_cities()) + 1,
    chromosome_generator=tsp_chromosome,
    selection_method="best",
    crossover_method="single_point",
    mutation_method="uniform",
)


def compute_distance_matrix(cities: dict) -> np.ndarray:
    """
    Compute the distance matrix between ALL cities.

    Parameters
    ----------
    cities : dict
        The cities with their coordinates.

    Returns
    -------
    np.ndarray
        The distance matrix between all cities
    """
    num_cities = len(cities)
    city_coords = np.array(list(cities.values()))
    dist_matrix = np.linalg.norm(city_coords[:, np.newaxis] - city_coords, axis=2)
    return dist_matrix

cities = set_cities()
dist_matrix = compute_distance_matrix(cities)
custom_args = [cities, dist_matrix]
ga.evolve(tsp_fitness_function, custom_args)
tsp_results(ga.best_chromosome)