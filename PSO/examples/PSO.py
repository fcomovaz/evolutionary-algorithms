import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Literal


class Particle:
    """
    A particle is a single solution in the search space of the optimization problem.
    Each particle has a position and a velocity which are updated in each iteration
    of the algorithm. The position of the particle is updated based on the velocity
    and the velocity is updated based on the position of the particle.

    Parameters
    ----------
    particle_dimension : np.uint16
        The number of dimensions of the particle
    min_value : np.float64
        The minimum value of the particle
    max_value : np.float64
        The maximum value of the particle
    dtype : np.dtype, optional
        The data type of the particle, by default np.float64
    limit_search_space : Literal[True, False], optional
        Whether to limit the search space or not, by default False

    Attributes
    ----------
    x : np.ndarray
        The position of the particle
    v : np.ndarray
        The velocity of the particle
    pbest : np.ndarray
        The best position of the particle
    fitness : np.float64
        The fitness of the particle

    Methods
    -------
    update_velocity(gbest, self_confidence_c1, swarm_confidence_c2, inertia_weight_w)
        Update the velocity of the particle based on the global best position
    update_position()
        Update the position of the particle based on the velocity
    evaluate(fitness_function)
        Evaluate the fitness of the particle based on the fitness function
    """

    def __init__(
        self,
        particle_dimension: np.uint16,
        min_value: np.float64,
        max_value: np.float64,
        dtype: np.dtype = np.float64,
        limit_search_space: Literal[True, False] = False,
    ):
        self.particle_dimension = particle_dimension
        self.min_value = min_value
        self.max_value = max_value
        self.dtype = dtype
        self.limit_search_space = limit_search_space
        self.x = (
            np.random.rand(particle_dimension) * (max_value - min_value) + min_value
        ).astype(dtype)
        self.v = np.random.rand(particle_dimension)
        self.pbest = self.x.copy()
        self.fitness = 0

    def update_velocity(
        self, gbest, self_confidence_c1, swarm_confidence_c2, inertia_weight_w
    ):
        self.v = (
            inertia_weight_w * self.v
            + self_confidence_c1 * np.random.rand() * (self.pbest - self.x)
            + swarm_confidence_c2 * np.random.rand() * (gbest - self.x)
        )

    def update_position(self):
        if self.limit_search_space:
            self.x = np.clip(self.x + self.v, self.min_value, self.max_value).astype(
                self.dtype
            )
        else:
            self.x = (self.x + self.v).astype(self.dtype)

    def evaluate(self, fitness_function: callable):
        self.fitness = fitness_function(self.x)


class PSO:
    """
    In computational science, particle swarm optimization is a
    computational method that optimizes a problem by iteratively trying
    to improve a candidate solution with regard to a given measure of quality.
    PSO is originally attributed to Kennedy, Eberhart and Shi and was first
    intended for simulating social behaviour, as a stylized representation
    of the movement of organisms in a bird flock or fish school.

    Parameters
    ----------
    paticle_size : np.uint16
        The number of particles in the swarm
    particle_dimension : np.uint16
        The number of dimensions of the particle
    min_value : np.float64
        The minimum value of the particle
    max_value : np.float64
        The maximum value of the particle
    fitness_function : callable
        The fitness function to be optimized
    self_confidence_c1 : np.float64, optional
        The self-confidence parameter, by default 0.5
    swarm_confidence_c2 : np.float64, optional
        The swarm-confidence parameter, by default 0.5
    inertia_weight_w : np.float64, optional
        The inertia weight parameter, by default 0.4
    iterations : np.uint16, optional
        The number of iterations, by default 100
    minimize : Literal[True, False], optional
        Whether to minimize or maximize the fitness function, by default True
    dtype : np.dtype, optional
        The data type of the particle, by default np.float64
    verbose : Literal[True, False], optional
        Whether to print the best fitness value for each iteration, by default False
    """

    def __init__(
        self,
        paticle_size: np.uint16,
        particle_dimension: np.uint16,
        min_value: np.float64,
        max_value: np.float64,
        fitness_function: callable,
        self_confidence_c1: np.float64 = 0.5,
        swarm_confidence_c2: np.float64 = 0.5,
        inertia_weight_w: np.float64 = 0.4,
        iterations: np.uint16 = 100,
        minimize: Literal[True, False] = True,
        dtype: np.dtype = np.float64,
        limit_search_space: Literal[True, False] = True,
        verbose: Literal[True, False] = False,
    ):

        # Initialize the parameters
        self.paticle_size = paticle_size
        self.particle_dimension = particle_dimension
        self.min_value = min_value
        self.max_value = max_value
        self.self_confidence_c1 = self_confidence_c1
        self.swarm_confidence_c2 = swarm_confidence_c2
        self.inertia_weight_w = inertia_weight_w
        self.iterations = iterations
        self.minimize = minimize
        self.fitness_function = fitness_function
        self.dtype = dtype
        self.limit_search_space = limit_search_space
        self.verbose = verbose

        # Initialize the particles using the Particle class and list comprehension
        # particles = []
        # for particle in range(paticle_size)
        #    particles.append(Particle(particle_dimension, min_value,
        #                              max_value, dtype, limit_search_space)
        #                     )
        # self.particles = particles
        self.particles = [
            Particle(
                particle_dimension,
                min_value,
                max_value,
                dtype,
                limit_search_space,
            )
            for _ in range(paticle_size)
        ]

        # Initialize the global best position and fitness value
        # gbeest is first particle position and fitness is 0 for convenience
        self.gbest = self.particles[0].x.copy()
        self.gbest_fitness = 0

        # Initialize the best fitness and position for each iteration
        self.best_fitness = np.zeros(iterations)
        self.best_position = np.zeros((iterations, particle_dimension))

    def optimize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Optimize the fitness function using the PSO algorithm

        Returns
        -------
        best_fitness : np.ndarray
            The best fitness value for each iteration
        best_position : np.ndarray
            The best position for each iteration
        """

        # Iterate over the number of iterations
        for iteration in range(self.iterations):
            if self.verbose:
                print(
                    f"\rIteration: {iteration}, Best Fitness: {self.gbest_fitness}",
                    end="",
                )
            # Iterate over the particles in the swarm
            for particle in self.particles:
                # 1. Update the velocity of the particle
                particle.update_velocity(
                    self.gbest,
                    self.self_confidence_c1,
                    self.swarm_confidence_c2,
                    self.inertia_weight_w,
                )

                # 2. Update the position of the particle
                particle.update_position()

                # 3. Evaluate the fitness of the particle
                particle.evaluate(self.fitness_function)

                # 4. Update the personal best and global best
                self.gbest_fitness = self.fitness_function(self.gbest)
                if self.minimize:
                    # if the fitness function is to be minimized
                    # then the fitness value should be less than the global best fitness
                    if particle.fitness < self.fitness_function(particle.pbest):
                        particle.pbest = particle.x.copy()
                    if particle.fitness < self.gbest_fitness:
                        self.gbest = particle.x.copy()
                else:
                    # if the fitness function is to be maximized
                    # then the fitness value should be greater than the global best fitness
                    if particle.fitness > self.fitness_function(particle.pbest):
                        particle.pbest = particle.x.copy()
                    if particle.fitness > self.gbest_fitness:
                        self.gbest = particle.x.copy()

            # Set the best fitness and position for each iteration (for history)
            self.best_fitness[iteration] = self.gbest_fitness
            self.best_position[iteration] = self.gbest.copy()

        return self.best_fitness, self.best_position

    def clean_memory(self):
        del self.particles
        del self.gbest
        del self.best_fitness
        del self.best_position

    def plot_fitness(self):
        plt.plot(self.best_fitness)
        plt.xlabel("iterationations")
        plt.ylabel("Fitness")
        plt.show()
