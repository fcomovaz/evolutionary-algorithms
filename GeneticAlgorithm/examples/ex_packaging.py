import matplotlib.pyplot as plt
from genetic_algorithm import *
import cv2
import os


def get_grid_vertices(array, separation):
    num_rows, num_cols = array.shape
    vertices = []

    for i in range(0, num_rows, separation):
        for j in range(0, num_cols, separation):
            vertices.append((j, i))

    x = [vertex[0] for vertex in vertices]
    y = [vertex[1] for vertex in vertices]

    return x, y


def get_grid_centers(x_grid, y_grid, separation, noise_level=0):
    # get the center of each pseudogrid
    xm = [int(x_grid[i] - separation / 2) for i in range(len(x_grid))]
    ym = [int(y_grid[i] - separation / 2) for i in range(len(y_grid))]

    # add noise to the coordinates
    moving = int(np.random.uniform(-noise_level, noise_level) * separation)
    for i in range(len(xm)):
        xm[i] += moving
        ym[i] += moving

        # check not overpass the max x_grid and y_grid
        if xm[i] >= x_grid[-1]:
            xm[i] = x_grid[-1]
        if ym[i] >= y_grid[-1]:
            ym[i] = y_grid[-1]

    # conver to a tuple
    coord = [(xm[i], ym[i]) for i in range(len(xm))]
    # remove the negative values
    coord = [
        coord[i] for i in range(len(coord)) if coord[i][0] >= 0 and coord[i][1] >= 0
    ]

    return coord


def packaging_chromosome() -> np.ndarray:
    """
    Create a chromosome.

    Returns
    -------
    np.ndarray
        The created chromosome.
    """

    pass


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


ga = GeneticAlgorithm(
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

# ga.evolve(my_optimization_function)
# ga.print_results()
# ga.plot_fitness(plt)


# Packing problem
pwd = os.path.dirname(os.path.realpath(__file__))
img = cv2.imread(pwd + "/gto.png", cv2.IMREAD_GRAYSCALE)
img = (img > 0).astype(np.uint8)
map_of_coordinates = img
map_shape = img.shape

overlapping_radius = 2
container_radius = 20
separation = int(container_radius * overlapping_radius)
x_grid, y_grid = get_grid_vertices(map_of_coordinates, separation)

# plot the grid above the map
plt.imshow(map_of_coordinates, cmap="gray")
plt.scatter(x_grid, y_grid, c="red", s=1)
plt.show()
