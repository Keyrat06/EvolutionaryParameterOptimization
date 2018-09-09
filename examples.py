import matplotlib.pyplot as plt
import numpy as np
import time

from evolutionary_optimization.evolutionary_parameter_optimization import EvolutionaryParameterOptimizer


def test_one_minimization():
    # Show solving the minimum of simple function
    scoring_function = lambda x, y: ((x + 2) ** 2 + (y + 2) ** 2) ** 0.5
    parameter_ranges = [(-20, 20), (-20, 20)]
    parameter_discretization = [0.1, 0.1]
    time_start = time.time()
    optimizer = EvolutionaryParameterOptimizer(
        parameter_ranges=parameter_ranges,
        parameter_discretization=parameter_discretization,
        fitness_function=scoring_function,
        population_size=10,
        generations=10,
        replacement_proportion=0.5,
        minimize=True,
        show_evolution=True)
    time_end = time.time()

    print("Our function is ((x+2)**2 + (y+2)**2)**0.5")
    print("Optimization runtime is {}".format(time_end - time_start))
    print("The optimizer found that x={0}, y={1} scored best for our function".format(*optimizer.best_individual))
    print("With this assignment we got a score of {}".format(optimizer.best_fitness))


def test_one_regression():
    # Show solving the values of a regression

    true_m = -5
    true_b = 5
    data_x = np.random.uniform(-10, 10, 50)
    data_y = true_m * data_x + true_b + np.random.normal(0, 5, data_x.shape)  # y = m * x + b + noise is what we will solve
    plt.scatter(data_x, data_y, label="data")
    plt.legend()
    plt.show()

    scoring_function = lambda m, b: sum(
        [(data_y[i] - (m * data_x[i] + b)) ** 2 for i in range(len(data_x))])  # Squared Distance

    parameter_ranges = [(-20, 20), (-20, 20)]
    parameter_discretization = [0.1, 0.1]
    time_start = time.time()
    optimizer = EvolutionaryParameterOptimizer(
        parameter_ranges=parameter_ranges,
        parameter_discretization=parameter_discretization,
        fitness_function=scoring_function,
        population_size=20,
        generations=20,
        replacement_proportion=0.5,
        minimize=True,
        show_evolution=True)
    #
    time_end = time.time()
    print("Our function is a regression with optimal m = -5, b = 5")
    print("optimization runtime is {}".format(time_end - time_start))
    print("The optimizer found that m={0}, b={1} scored best for our function".format(*optimizer.best_individual))
    print("With this assignment we got a score of {}".format(optimizer.best_fitness))

    x_out = np.linspace(-10, 10, 100)
    y_out = optimizer.best_individual[0] * x_out + optimizer.best_individual[1]
    plt.clf()
    plt.scatter(data_x, data_y, c='b', label="data")
    plt.plot(x_out, y_out, c='r', label="guessed_line")
    plt.legend()
    plt.show("hold")


if __name__ == '__main__':
    test_one_minimization()
    time.sleep(0.5)
    test_one_regression()
