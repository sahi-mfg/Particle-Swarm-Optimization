import unittest
import numpy as np
from pso import PSO


class TestPSO(unittest.TestCase):
    def setUp(self):
        self.pso = PSO(fonction=lambda x, y: x**2 + y**2, particle=2, iteration=50, dim=2, min=-10, max=10)

    def test_eval_fonction(self):
        self.pso.position = np.array([[3, 4], [5, 12]])
        expected_fitness = np.array([25, 169])
        np.testing.assert_array_equal(self.pso.eval_fonction(), expected_fitness)

    def test_update_personal_best(self):
        self.pso.position = np.array([[3, 4], [5, 12]])
        self.pso.fitness = np.array([25, 169])
        self.pso.personal_best_fitness = np.array([30, 200])
        self.pso.update_personal_best()
        np.testing.assert_array_equal(self.pso.personal_best_position, self.pso.position)
        np.testing.assert_array_equal(self.pso.personal_best_fitness, self.pso.fitness)

    def test_update_personal_best_only_improved_particles(self):
        # particule 0 s'améliore, particule 1 se dégrade : seule la 0 doit changer
        self.pso.personal_best_position = np.array([[1.0, 1.0], [0.0, 0.0]])
        self.pso.personal_best_fitness = np.array([2.0, 0.0])
        self.pso.position = np.array([[0.5, 0.5], [9.0, 9.0]])
        self.pso.fitness = np.array([0.5, 162.0])
        self.pso.update_personal_best()
        np.testing.assert_array_equal(self.pso.personal_best_position, [[0.5, 0.5], [0.0, 0.0]])
        np.testing.assert_array_equal(self.pso.personal_best_fitness, [0.5, 0.0])

    def test_update_personal_best_does_not_alias_current_state(self):
        self.pso.position = np.array([[3.0, 4.0], [5.0, 12.0]])
        self.pso.fitness = np.array([25.0, 169.0])
        self.pso.personal_best_fitness = np.array([30.0, 200.0])
        self.pso.update_personal_best()
        self.pso.position[0, 0] = 99.0
        self.pso.fitness[0] = 99.0
        self.assertEqual(self.pso.personal_best_position[0, 0], 3.0)
        self.assertEqual(self.pso.personal_best_fitness[0], 25.0)

    def test_update_global_best_uses_personal_best_position(self):
        # la meilleure fitness personnelle (0) a été trouvée en (0, 0), pas à la position courante
        self.pso.personal_best_position = np.array([[0.0, 0.0], [5.0, 5.0]])
        self.pso.personal_best_fitness = np.array([0.0, 50.0])
        self.pso.position = np.array([[9.0, 9.0], [5.0, 5.0]])
        self.pso.global_best_fitness = np.inf
        self.pso.update_global_best()
        np.testing.assert_array_equal(self.pso.global_best_position, [0.0, 0.0])
        self.assertEqual(self.pso.global_best_fitness, 0.0)

    def test_run_converges_on_sphere(self):
        np.random.seed(0)
        pso = PSO(fonction=lambda x, y: x**2 + y**2, particle=30, iteration=100, dim=2, min=-10, max=10, w=0.7, phi1=1.5, phi2=1.5)
        position, fitness = pso.run()
        self.assertLess(fitness, 1e-6)
        self.assertAlmostEqual(fitness, position[0] ** 2 + position[1] ** 2)


if __name__ == "__main__":
    unittest.main()
