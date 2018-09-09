import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tqdm import tqdm


class EvolutionaryParameterOptimizer:
    """
    Evolutionary Algorithm Optimizer for arbitrary parameter_discretization which allows it to solve integer optimizations
    """

    def __init__(self, parameter_ranges, parameter_discretization, fitness_function, population_size=20,
                 replacement_proportion=0.5, generations=20, minimize=False, show_evolution=False,
                 random_variation_probability=0.1, crossover_probability=0.7):
        self.minimize = minimize
        self.show_evolution = show_evolution
        if self.show_evolution:
            plt.ion()
            self.cb = None

        self.population_size = population_size
        assert 0 < self.population_size
        self.replacement_proportion = replacement_proportion
        assert 0.0 < self.replacement_proportion <= 1.0
        self.parameter_ranges = np.array(parameter_ranges)
        self.parameter_discretization = np.array(parameter_discretization)
        assert len(self.parameter_ranges) == len(self.parameter_discretization)
        self.fitness_function = fitness_function
        self.best_individual = None
        self.best_fitness = float("inf") if self.minimize else -float("inf")

        self.fitnesses = np.zeros(self.population_size)
        self.parameter_interval = self.parameter_ranges[:, 1] - self.parameter_ranges[:, 0]
        self.population = np.random.random((self.population_size, len(self.parameter_ranges))) *\
                          self.parameter_interval + self.parameter_ranges[:, 0]
        self.population -= (self.population % self.parameter_discretization)
        self.generations = generations
        assert 0 < self.generations
        self.random_variation_probability = random_variation_probability
        assert 0.0 <= self.random_variation_probability <= 1.0
        self.crossover_probability = self.random_variation_probability + crossover_probability
        assert 0.0 <= self.crossover_probability <= 1.0
        self.fitness_cache = dict()
        self.best_fitnesses = []

        self.optimize()

    def evaluate_fitness(self):
        fitnesses = []
        for individual in self.population:
            if tuple(individual) in self.fitness_cache:
                fitness = self.fitness_cache[tuple(individual)]
            else:
                fitness = self.fitness_function(*individual)
                self.fitness_cache[tuple(individual)] = fitness

            fitnesses.append(fitness)

            if (self.minimize and fitness < self.best_fitness) or (not self.minimize and fitness > self.best_fitness):
                self.best_fitness = fitness
                self.best_individual = individual

        self.fitnesses = np.array(fitnesses)
        self.best_fitnesses.append(self.best_fitness)

    def replacement(self):
        population_fitness_variation = self.get_population_fitness_variation(noise_level=0.01)
        survivors_inx = sorted(range(self.population_size),
                               key=lambda x: population_fitness_variation[x],
                               reverse=True)[0:int(self.population_size * self.replacement_proportion)]
        self.population = self.population[survivors_inx]
        self.fitnesses = self.fitnesses[survivors_inx]

    def get_population_fitness_variation(self, noise_level=0.0):
        population_fitness_variation = -1 * self.fitnesses if self.minimize else self.fitnesses
        population_fitness_variation += np.random.normal(0, noise_level, population_fitness_variation.shape)
        population_fitness_variation -= population_fitness_variation.min()
        return population_fitness_variation

    def population_variation(self):
        population_fitness_variation = self.get_population_fitness_variation(noise_level=0.01)
        new_population = list(self.population)
        total_fitness_variation = sum(population_fitness_variation)
        _selection_weights = (population_fitness_variation / total_fitness_variation)

        while len(new_population) < self.population_size:
            rnd = np.random.random()
            if rnd < self.random_variation_probability:  # go random (random_variation_probability of time)
                child = np.random.random((len(self.parameter_ranges))) * self.parameter_interval + self.parameter_ranges[:, 0]
                child -= (child % self.parameter_discretization)
            elif rnd < self.crossover_probability:  # sexual reproduction (crossover_probability of time)
                father_index, mother_index = np.random.choice(range(len(self.population)), 2,
                                                              replace=False, p=_selection_weights)
                father, mother = self.population[father_index], self.population[mother_index]
                child = (father + mother) / 2
                child -= (child % self.parameter_discretization)
            else:  # asexual reproduction (rest of time)
                parent_index = np.random.choice(range(len(self.population)), 1,
                                                replace=False, p=_selection_weights)[0]
                parent = self.population[parent_index]
                child = []
                for i in range(len(parent)):
                    s = int(np.std(self.population[:, i]) + 1)
                    d = s * int(np.random.normal(0, 10)) * self.parameter_discretization[i]
                    child_param = parent[i] + d
                    child_param = min(self.parameter_ranges[i][1], child_param)
                    child_param = max(self.parameter_ranges[i][0], child_param)
                    child.append(child_param)
            new_population.append(np.array(child))
        self.population = np.array(new_population)

    def optimize(self):
        for _ in tqdm(range(self.generations)):
            self.population_variation()
            self.evaluate_fitness()
            if self.show_evolution:
                self.show()
            self.replacement()

    def show(self):
        if len(self.best_individual) == 2:
            plt.cla()
            xs = self.population
            x, y = xs[:, 0], xs[:, 1]
            z = self.fitnesses
            sc = plt.scatter(x, y, c=z, marker='o', cmap=cm.jet, label="all_fitnesses")
            if self.best_individual is not None:
                plt.scatter(self.best_individual[0], self.best_individual[1], c='r', marker='^', label="best_fitness")
            if self.cb is None:
                self.cb = plt.colorbar(sc)
            plt.xlim(*self.parameter_ranges[0])
            plt.ylim(*self.parameter_ranges[1])
            plt.pause(0.00001)
        else:
            plt.cla()
            plt.plot(self.best_fitnesses)
            plt.pause(0.00001)

