import numpy as np
from typing import Union, Literal


class GA:
    """
    Genetic Algorithm class for optimization problems.
    ------

    The genetic algorithm is a search heuristic that mimics the process of natural
    selection. This heuristic is routinely used to generate useful solutions to
    optimization and search problems. It was first introduced by John Holland in 1960.
    """

    def __init__(
        self,
        generations: np.uint16,
        population_size: np.uint16,
        chromosome_length: np.uint16,
        chromosome_min_value: Union[np.int16, np.float32] = 0,
        chromosome_max_value: Union[np.int16, np.float32] = 5,
        chromosome_dtype: type = np.int16,
        chromosome_generator: callable = None,
        mutation_probability: np.float32 = 0.1,
        crossover_probability: np.float32 = 0.5,
        optimization_type: Literal["minimize", "maximize"] = "minimize",
        selection_method: Literal["best", "random", "rank"] = "best",
        mutation_method: Literal["slight", "uniform", "arithmetical"] = "slight",
        crossover_method: Literal[
            "mean", "single_point", "two_point", "uniform", "swap"
        ] = "single_point",
    ):
        """
        Parameters
        ----------
        generations : int
            The number of generations to run the genetic algorithm.
        population_size : int
            The number of individuals in the population.
        chromosome_length : int
            The length of the chromosome.
        chromosome_min_value : Union[int, np.float32], optional
            The minimum value of the chromosome. Default is 0.
        chromosome_max_value : Union[int, np.float32], optional
            The maximum value of the chromosome. Default is 5.
        chromosome_dtype : type, optional
            The data type of the chromosome. Default is np.int8.
        mutation_probability : np.float32, optional
            The mutation rate. Default is 0.01.
        crossover_probability : np.float32, optional
            The crossover rate. Default is 0.5.
        elitism_count : int, optional
            The number of elite individuals to be passed to the next generation.
            Default is 4.
        optimization_type : str, optional
            The type of optimization (minimize or maximize). Default is "minimize".
        selection_method : str, optional
            The selection method (best, random, rank). Default is "best".

        Returns
        -------
        GA: object
            The GA object.
        """

        # Ensure deterministic results
        np.random.seed(42)

        # Genetic Algorithm Parameter
        self.generations = generations
        self.population_size = population_size

        self.chromosome_length = chromosome_length
        self.chromosome_min_value = chromosome_min_value
        self.chromosome_max_value = chromosome_max_value
        self.chromosome_dtype = chromosome_dtype
        self.chromosome_generator = chromosome_generator

        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.elitism_count = self.population_size - 2

        self.optimization_type = optimization_type
        self.selection_method = selection_method
        self.mutation_method = mutation_method
        self.encode = crossover_method


        # Variables when the genetic algorithm is end
        self.best_chromosome = None
        self.best_fitness = None
        self.historical_fitness = np.zeros(self.generations)

    def init_population(self) -> np.ndarray:
        """
        Initialize the population.

        Returns
        -------
        np.ndarray
            The initial population of individuals.
        """

        population = []
        for _ in range(self.population_size):
            if self.chromosome_generator is None:
                chromosome = self._create_chromosome()
            else:
                chromosome = self.chromosome_generator()
            population.append(chromosome)
        return population

    def _create_chromosome(self) -> np.ndarray:
        """
        Create a chromosome.

        Returns
        -------
        np.ndarray
            The created chromosome.
        """

        if np.issubdtype(self.chromosome_dtype, np.integer):
            chromosome = np.random.randint(
                self.chromosome_min_value,
                self.chromosome_max_value,
                self.chromosome_length,
            ).astype(self.chromosome_dtype)
        else:
            chromosome = np.random.uniform(
                self.chromosome_min_value,
                self.chromosome_max_value,
                self.chromosome_length,
            ).astype(self.chromosome_dtype)
        return chromosome

    def calculate_fitness(
        self, population: np.ndarray, fitness_function=None, custom_args: list = None
    ) -> np.ndarray:
        """
        Calculate the fitness of the population.

        Parameters
        ----------
        population : np.ndarray
            The population of individuals.
        fitness_function : function, optional
            The fitness function to be optimized. Default is None.

        Returns
        -------
        np.ndarray
            The fitness scores of the population
        """

        fitness_scores = []
        for chromosome in population:
            fitness_scores.append(fitness_function(chromosome, custom_args))
        return np.array(fitness_scores)

    def select_parents(
        self,
        population: np.ndarray,
        fitness_scores: np.ndarray,
        optimization_type: str = "minimize",
    ) -> np.ndarray:
        """
        Select parents for crossover.

        Parameters
        ----------
        population : np.ndarray
            The population of individuals.
        fitness_scores : np.ndarray
            The fitness scores of the population.
        optimization_type : str, optional
            The type of optimization (minimize or maximize). Default is "minimize".

        Returns
        -------
        np.ndarray
            The selected parents for crossover.
        """

        if optimization_type not in ["minimize", "maximize"]:
            raise ValueError(
                "Optimization type must be either 'minimize' or 'maximize'."
            )

        parents = np.zeros((2, self.chromosome_length), dtype=self.chromosome_dtype)

        if self.selection_method == "best":
            parents = self._best_selection(population, fitness_scores)

        if self.selection_method == "random":
            parents = self._random_selection(population)

        if self.selection_method == "rank":
            parents = self._rank_selection(population, fitness_scores)

        return parents

    def _best_selection(
        self, population: np.ndarray, fitness_scores: np.ndarray
    ) -> np.ndarray:
        """
        Select parents for crossover.

        Parameters
        ----------
        population : np.ndarray
            The population of individuals.
        fitness_scores : np.ndarray
            The fitness scores of the population.

        Returns
        -------
        np.ndarray
            The selected parents for crossover.
        """

        parents = np.zeros((2, self.chromosome_length), dtype=self.chromosome_dtype)

        # Sort fitness scores and get the indexes for selection
        if self.optimization_type == "maximize":
            sorted_indexes = np.argsort(fitness_scores)[::-1]
        else:  # "minimize"
            sorted_indexes = np.argsort(fitness_scores)

        # Select the best two individuals as parents
        parents[0] = population[sorted_indexes[0]]
        parents[1] = population[sorted_indexes[1]]

        return parents

    def _random_selection(self, population: np.ndarray) -> np.ndarray:
        """
        Select parents for crossover.

        Parameters
        ----------
        population : np.ndarray
            The population of individuals.

        Returns
        -------
        np.ndarray
            The selected parents for crossover.
        """
        parents = np.zeros((2, self.chromosome_length), dtype=self.chromosome_dtype)
        indexes = np.random.choice(range(self.population_size), 2, replace=False)
        parents[0] = population[indexes[0]]
        parents[1] = population[indexes[1]]

        return parents

    def _rank_selection(
        self, population: np.ndarray, fitness_scores: np.ndarray
    ) -> np.ndarray:
        """
        Select parents for crossover.

        Parameters
        ----------
        population : np.ndarray
            The population of individuals.
        fitness_scores : np.ndarray
            The fitness scores of the population.

        Returns
        -------
        np.ndarray
            The selected parents for crossover.
        """

        parents = np.zeros((2, self.chromosome_length), dtype=self.chromosome_dtype)

        # Rank the fitness scores
        if self.optimization_type == "maximize":
            rank = np.argsort(np.argsort(-fitness_scores))
        else:  # "minimize"
            rank = np.argsort(np.argsort(fitness_scores))

        # Calculate the selection probability
        selection_probability = (
            2 - self.population_size
        ) / self.population_size + 2 * (rank - 1) * (
            2 - (2 - self.population_size) / self.population_size
        ) / (
            self.population_size * (self.population_size - 1)
        )
        selection_probability = np.abs(selection_probability) / np.sum(
            np.abs(selection_probability)
        )
        # Select the parents
        indexes = np.random.choice(
            range(self.population_size), 2, p=selection_probability
        )
        parents[0] = population[indexes[0]]
        parents[1] = population[indexes[1]]

        return parents

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform crossover between two parents.

        Parameters
        ----------
        parent1 : np.ndarray
            The first parent.
        parent2 : np.ndarray
            The second parent.

        Returns
        -------
        np.ndarray
            The offspring of the parents.
        """

        # if a random number is greater than the crossover rate, perform crossover
        if np.random.rand() < self.crossover_probability:
            if self.encode == "mean" or self.chromosome_length == 1:
                return self._mean_crossover(parent1, parent2)

            if self.encode == "single_point":
                return self._single_point_crossover(parent1, parent2)

            if self.chromosome_length > 2:
                if self.encode == "two_point":
                    return self._two_point_crossover(parent1, parent2)
                if self.encode == "uniform":
                    return self._uniform_crossover(parent1, parent2)
                if self.encode == "swap":
                    return self._swap_crossover(parent1, parent2)

        # Crossover not performed
        return np.array([parent1, parent2])

    def _mean_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform mean crossover between two parents.

        Parameters
        ----------
        parent1 : np.ndarray
            The first parent.
        parent2 : np.ndarray
            The second parent.

        Returns
        -------
        np.ndarray
            The offspring of the parents.
        """

        off1 = np.mean([parent1, parent2], axis=0)
        off2 = np.mean([parent1, parent2], axis=0)
        offspring = np.array([off1, off2]).astype(self.chromosome_dtype)
        return offspring

    def _single_point_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> np.ndarray:
        """
        Perform single-point crossover between two parents.

        Parameters
        ----------
        parent1 : np.ndarray
            The first parent.
        parent2 : np.ndarray
            The second parent.

        Returns
        -------
        np.ndarray
            The offspring of the parents.
        """

        crossover_point = np.random.randint(1, self.chromosome_length)
        off_1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        off_2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        offspring = np.array([off_1, off_2]).astype(self.chromosome_dtype)
        return offspring

    def _two_point_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> np.ndarray:
        """
        Perform two-point crossover between two parents.

        Parameters
        ----------
        parent1 : np.ndarray
            The first parent.
        parent2 : np.ndarray
            The second parent.

        Returns
        -------
        np.ndarray
            The offspring of the parents.
        """

        crossover_points = np.sort(
            np.random.choice(self.chromosome_length, 2, replace=False)
        )
        off_1 = np.concatenate(
            (
                parent1[: crossover_points[0]],
                parent2[crossover_points[0] : crossover_points[1]],
                parent1[crossover_points[1] :],
            )
        )
        off_2 = np.concatenate(
            (
                parent2[: crossover_points[0]],
                parent1[crossover_points[0] : crossover_points[1]],
                parent2[crossover_points[1] :],
            )
        )
        offspring = np.array([off_1, off_2]).astype(self.chromosome_dtype)
        return offspring

    def _uniform_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> np.ndarray:
        """
        Perform uniform crossover between two parents.

        Parameters
        ----------
        parent1 : np.ndarray
            The first parent.
        parent2 : np.ndarray
            The second parent.

        Returns
        -------
        np.ndarray
            The offspring of the parents.
        """

        mask = np.random.randint(0, 2, self.chromosome_length)
        off_1 = np.where(mask, parent1, parent2)
        off_2 = np.where(mask, parent2, parent1)
        offspring = np.array([off_1, off_2])
        return offspring

    def _swap_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform swap crossover between two parents.

        Parameters
        ----------
        parent1 : np.ndarray
            The first parent.
        parent2 : np.ndarray
            The second parent.

        Returns
        -------
        np.ndarray
            The offspring of the parents.
        """

        mask = np.arange(self.chromosome_length) % 2
        off_1 = np.where(mask, parent1, parent2)
        off_2 = np.where(mask, parent2, parent1)
        offspring = np.array([off_1, off_2])
        return offspring

    def mutate(
        self,
        offspring: np.ndarray,
    ) -> np.ndarray:
        """
        Perform mutation on the chromosome.

        Parameters
        ----------
        chromosome : np.ndarray
            The offspring to be mutated.

        Returns
        -------
        np.ndarray
            The mutated chromosome.
        """

        if np.random.rand() < self.mutation_probability:
            if self.mutation_method not in ["slight", "uniform", "arithmetical"]:
                raise ValueError(
                    "Mutation type must be either 'slight', 'uniform' or 'arithmetical'."
                )

            if self.chromosome_length == 1:
                if self.mutation_method == "slight":
                    offspring = self._slight_mutation(offspring)
                if self.mutation_method == "uniform":
                    offspring = self._uniform_mutation(offspring)
                if self.mutation_method == "arithmetical":
                    offspring = self._arithmetical_mutation(offspring)
                return offspring

            else:

                for i in range(len(offspring)):
                    if self.mutation_method == "slight":
                        offspring[i] = self._slight_mutation(offspring[i])
                    if self.mutation_method == "uniform":
                        offspring[i] = self._uniform_mutation(offspring[i])
                    if self.mutation_method == "arithmetical":
                        offspring[i] = self._arithmetical_mutation(offspring[i])

        # Mutation not performed
        return offspring

    def _slight_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Perform slight mutation on the chromosome.

        Parameters
        ----------
        chromosome : np.ndarray
            The chromosome to be mutated.

        Returns
        -------
        np.ndarray
            The mutated chromosome.
        """

        mask = np.random.choice([0, 1], self.chromosome_length, p=[0.1, 0.9])
        if self.chromosome_generator is not None:
            chromosome = np.where(mask, chromosome, self.chromosome_generator())
        else:
            chromosome = np.where(mask, chromosome, self._create_chromosome())
        return chromosome

    def _uniform_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Perform swap mutation on the chromosome.

        Parameters
        ----------
        chromosome : np.ndarray
            The chromosome to be mutated.

        Returns
        -------
        np.ndarray
            The mutated chromosome.
        """

        mask = np.random.randint(0, 2, self.chromosome_length)
        chromosome = np.where(mask, chromosome, np.roll(chromosome, 1))
        return chromosome

    def _arithmetical_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Perform arithmetical mutation on the chromosome.

        Parameters
        ----------
        chromosome : np.ndarray
            The chromosome to be mutated.

        Returns
        -------
        np.ndarray
            The mutated chromosome.
        """

        if np.issubdtype(self.chromosome_dtype, np.integer):
            mutation = np.random.randint(-1, 1, self.chromosome_length)
        if np.issubdtype(self.chromosome_dtype, np.floating):
            mutation = np.random.uniform(-0.1, 0.1, self.chromosome_length)

        mutation = mutation.astype(self.chromosome_dtype)
        chromosome = chromosome + mutation
        return chromosome

    def evolve(
        self,
        fitness_function: callable,
        custom_args: list = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Evolve the population.

        Parameters
        ----------
        fitness_function : callable
            The fitness function to be optimized.
        custom_args : list, optional
            The custom arguments for the fitness function. Default is None.
        verbose : bool, optional


        Returns
        -------
        np.ndarray
            The final population.
        """

        population = self.init_population()
        fitness_scores = self.calculate_fitness(
            population, fitness_function, custom_args
        )
        for generation in range(self.generations):
            if verbose:
                print(
                    f"Generation {generation+1}, Best Fitness: {np.min(fitness_scores)}"
                )
            # if generation < 2:
            #     for p in population:
            #         print(p)
            #     print()
            self.historical_fitness[generation] = np.min(fitness_scores)
            parents = self.select_parents(
                population, fitness_scores, optimization_type=self.optimization_type
            )
            offspring = self.crossover(parents[0], parents[1])
            mutated = self.mutate(offspring)
            population = self.elitism(population, fitness_scores, mutated)

            fitness_scores = self.calculate_fitness(
                population, fitness_function, custom_args
            )
        # return population
        self.select_best_chromosome(population, fitness_scores)

    def elitism(
        self, population: np.ndarray, fitness_scores: np.ndarray, mutated: np.ndarray
    ) -> np.ndarray:
        """
        Select the elite individuals from the population.

        Parameters
        ----------
        population : np.ndarray
            The population of individuals.
        fitness_scores : np.ndarray
            The fitness scores of the population.
        mutated : np.ndarray
            The mutated offspring.

        Returns
        -------
        np.ndarray
            The elite individuals.
        """

        # elite_indexes = np.argsort(fitness_scores)[: self.elitism_count]
        if self.optimization_type == "maximize":
            elite_indexes = np.argsort(fitness_scores)[-self.elitism_count :]
        else:  # "minimize"
            elite_indexes = np.argsort(fitness_scores)[: self.elitism_count]
        elite = np.array([population[i] for i in elite_indexes])
        population = np.concatenate((elite, mutated), axis=0)
        return population

    def select_best_chromosome(
        self, population: np.ndarray, fitness_scores: np.ndarray
    ) -> np.ndarray:
        """
        Select the best chromosome from the population.

        Parameters
        ----------
        population : np.ndarray
            The population of individuals.
        fitness_scores : np.ndarray
            The fitness scores of the population.

        Returns
        -------
        np.ndarray
            The best chromosome.
        """

        self.best_chromosome = population[np.argmin(fitness_scores)]
        self.best_fitness = np.min(fitness_scores)

    def print_results(self):
        """
        Print the best chromosome and fitness score.
        """

        print(f"Best Chromosome: {self.best_chromosome}")
        print(f"Best Fitness: {self.best_fitness}")

    def plot_fitness(self, plt: callable):
        """
        Plot the historical fitness of the genetic algorithm.
        """

        plt.plot(self.historical_fitness)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Genetic Algorithm")
        plt.show()
