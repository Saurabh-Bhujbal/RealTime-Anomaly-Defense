import numpy as np
from scipy.special import gamma

class CuckooSearchOptimizer:
    def __init__(self, fitness_fn, bounds, population_size=20, max_iter=30, pa=0.25, alpha=0.01):
        self.fitness_fn = fitness_fn
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.pa = pa          # fraction of worst nests abandoned each iteration
        self.alpha = alpha    # step size scaling
        self.dim = len(bounds)

    def _levy_flight(self):
        # Mantegna's algorithm for Levy-stable steps, beta=1.5
        beta = 1.5
        sigma_u = (gamma(1+beta) * np.sin(np.pi*beta/2) /
                   (gamma((1+beta)/2) * beta * 2**((beta-1)/2))) ** (1/beta)
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v) ** (1 / beta))

    def run(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        nests = low + np.random.rand(self.pop_size, self.dim) * (high - low)
        fitness = np.array([self.fitness_fn(n) for n in nests])
        best_idx = np.argmax(fitness)
        best_nest = nests[best_idx].copy()
        best_score = fitness[best_idx]
        history = [best_score]

        for t in range(1, self.max_iter + 1):
            # Generate new solutions via Levy flight from each nest
            for i in range(self.pop_size):
                step = self._levy_flight()
                new_nest = nests[i] + self.alpha * step * (nests[i] - best_nest)
                new_nest = np.clip(new_nest, low, high)
                new_score = self.fitness_fn(new_nest)
                if new_score > fitness[i]:
                    nests[i] = new_nest
                    fitness[i] = new_score

            # Abandon a fraction pa of the worst nests, replace randomly
            n_abandon = int(self.pa * self.pop_size)
            worst_idx = np.argsort(fitness)[:n_abandon]
            for idx in worst_idx:
                nests[idx] = low + np.random.rand(self.dim) * (high - low)
                fitness[idx] = self.fitness_fn(nests[idx])

            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_score:
                best_score = fitness[gen_best_idx]
                best_nest = nests[gen_best_idx].copy()
            history.append(best_score)

        return best_nest, best_score, history
