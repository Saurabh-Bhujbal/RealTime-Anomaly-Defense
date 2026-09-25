import numpy as np

class SalpSwarmOptimizer:
    def __init__(self, fitness_fn, bounds, population_size=20, max_iter=30):
        # fitness_fn(params: np.ndarray) -> float   (HIGHER is better)
        # bounds: list of (low, high) tuples, one per dimension
        self.fitness_fn = fitness_fn
        self.bounds = np.array(bounds)          # shape (D, 2)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.dim = len(bounds)

    def _init_population(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return low + np.random.rand(self.pop_size, self.dim) * (high - low)

    def run(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        salps = self._init_population()
        fitness = np.array([self.fitness_fn(s) for s in salps])
        best_idx = np.argmax(fitness)
        food = salps[best_idx].copy()
        best_score = fitness[best_idx]
        history = [best_score]

        for t in range(1, self.max_iter + 1):
            c1 = 2 * np.exp(-(4 * t / self.max_iter) ** 2)  # decays explore->exploit

            for i in range(self.pop_size):
                if i == 0:
                    # LEADER salp update (standard SSA leader equation)
                    c2 = np.random.rand(self.dim)
                    c3 = np.random.rand(self.dim)
                    step = c1 * ((high - low) * c2 + low)
                    salps[i] = np.where(c3 < 0.5, food + step, food - step)
                else:
                    # FOLLOWER salp update (Newton's law of motion, avg of neighbors)
                    salps[i] = (salps[i] + salps[i - 1]) / 2.0

                salps[i] = np.clip(salps[i], low, high)

            fitness = np.array([self.fitness_fn(s) for s in salps])
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_score:
                best_score = fitness[gen_best_idx]
                food = salps[gen_best_idx].copy()
            history.append(best_score)

        return food, best_score, history
