# Author: Sahi Gonsangbeu


import numpy as np
from tqdm import tqdm


class PSO:
    """
    Particle Swarm Optimization (PSO) class.

    This class implements the PSO algorithm, a computational method that optimizes a problem
    by iteratively trying to improve a candidate solution with regard to a given measure of quality.

    Attributes
    ----------
    fonction : function
        The function to be optimized.
    w : float
        Inertia weight.
    phi1, phi2 : float
        Acceleration coefficients.
    particle : int
        The number of particles in the swarm.
    dim : int
        The dimensionality of the problem space.
    iteration : int
        The number of iterations.
    min, max : array
        The lower and upper bounds of the problem space.
    position : array
        The current positions of all particles.
    velocity : array
        The current velocities of all particles.
    fitness : array
        The current fitness values of all particles.
    personal_best_position : array
        The personal best positions of all particles.
    personal_best_fitness : array
        The personal best fitness values of all particles.
    global_best_position : array
        The global best position.
    global_best_fitness : float
        The global best fitness value.
    global_best_fitness_history : list
        A list storing the global best fitness value at each iteration.
    coord : dict
        A dictionary storing the positions of all particles at each iteration.
    """

    def __init__(self, fonction, dim=None, particle=50, iteration=120, min=-100, max=100, w=0.8, phi1=2, phi2=2):
        self.fonction = fonction
        self.w = w  # inertie
        self.phi1, self.phi2 = phi1, phi2  # coeff d'acceleration
        self.particle = particle
        self.dim = dim
        self.iteration = iteration  # nombre itération

        self.min, self.max = np.array(min) * np.ones(self.dim), np.array(max) * np.ones(self.dim)

        self.position = np.random.uniform(low=self.min, high=self.max, size=(self.particle, self.dim))
        v_max = self.max - self.min
        self.velocity = np.random.uniform(low=-v_max, high=v_max, size=(self.particle, self.dim))
        self.fitness = self.eval_fonction()
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = self.fitness.copy()
        self.global_best_position = np.zeros(self.dim)
        self.global_best_fitness = np.inf
        self.global_best_fitness_history = []
        self.update_global_best()

        self.coord = {"X": []}

    def eval_fonction(self):
        """Evaluates the fitness function.

        Returns
        -------
        np.array
            the fitness values of all particles.
        """
        self.fitness = np.zeros(self.particle)
        for i in range(self.particle):
            self.fitness[i] = self.fonction(self.position[i][0], self.position[i][1])
        return self.fitness

    def update_personal_best(self):
        """updates the personal best positions and fitness values."""
        self.condition = self.personal_best_fitness > self.fitness
        # seules les particules qui se sont améliorées mettent à jour leur meilleure position
        self.personal_best_position[self.condition] = self.position[self.condition]
        self.personal_best_fitness[self.condition] = self.fitness[self.condition]

    def update_global_best(self):
        """updates the global best position and fitness value."""
        i = self.personal_best_fitness.argmin()
        if self.global_best_fitness > self.personal_best_fitness[i]:
            self.global_best_position = self.personal_best_position[i, :].copy()
            self.global_best_fitness = self.personal_best_fitness[i]

    def update_position(self):
        """updates the positions of all particles."""
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, self.min, self.max)

    def update_velocity(self):
        """updates the velocities of all particles."""
        a = np.random.rand(self.particle, self.dim)
        b = np.random.rand(self.particle, self.dim)
        self.velocity = (
            self.w * self.velocity
            + self.phi1 * a * (self.personal_best_position - self.position)
            + self.phi2 * b * (self.global_best_position - self.position)
        )

    def add(self):
        """adds the current positions of all particles to the coord dictionary."""
        self.coord["X"].append(self.position)

    def run(self):
        """Runs the PSO algorithm for a given number of iterations.

        Returns
        -------
        np.array
            best position.
        float
            best fitness value.
        """
        for iteration_no in tqdm(range(self.iteration)):
            self.add()
            self.update_velocity()
            self.update_position()
            self.eval_fonction()
            self.update_personal_best()
            self.update_global_best()

            print(f"Iter: {iteration_no}, Best fit: {self.global_best_fitness} at {self.global_best_position}")

            self.global_best_fitness_history.append(self.global_best_fitness)

        self.best_position, self.best_fitness = self.global_best_position, self.global_best_fitness
        return self.best_position, self.best_fitness
