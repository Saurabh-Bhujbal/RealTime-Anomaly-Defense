import numpy as np

class OspreyOptimizer:
    def __init__(self, fitness_fn, bounds, population_size=20, max_iter=30):
        # fitness_fn(params: np.ndarray) -> float   (HIGHER is better)
        # bounds: list of (low, high) tuples, one per dimension
        self.fitness_fn = fitness_fn
        self.bounds = np.array(bounds)
        self.pop_size = population_size
        self.max_iter = max_iter
        self.dim = len(bounds)

    def run(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]

        # Initialize population and evaluate
        ospreys = low + np.random.rand(self.pop_size, self.dim) * (high - low)
        fitness = np.array([self.fitness_fn(o) for o in ospreys])

        best_idx = np.argmax(fitness)
        best_osprey = ospreys[best_idx].copy()
        best_score = fitness[best_idx]
        history = [best_score]

        for t in range(1, self.max_iter + 1):

            for i in range(self.pop_size):

                # -----------------------------------------------
                # PHASE 1: Identify and attack fish (exploration)
                # -----------------------------------------------
                # "Fish position" = a randomly chosen population member
                # whose fitness is >= this osprey's fitness (falls back
                # to the global best if no such member exists yet).
                better_mask = fitness >= fitness[i]
                better_indices = np.where(better_mask)[0]
                if len(better_indices) > 0:
                    fish_idx = np.random.choice(better_indices)
                    fish_position = ospreys[fish_idx]
                else:
                    fish_position = best_osprey

                I = np.round(1 + np.random.rand())  # I is randomly 1 or 2
                r1 = np.random.rand(self.dim)
                candidate_p1 = ospreys[i] + r1 * (fish_position - I * ospreys[i])
                candidate_p1 = np.clip(candidate_p1, low, high)

                score_p1 = self.fitness_fn(candidate_p1)
                if score_p1 > fitness[i]:
                    ospreys[i] = candidate_p1
                    fitness[i] = score_p1

                # -----------------------------------------------
                # PHASE 2: Carry fish to suitable position (exploitation)
                # -----------------------------------------------
                # Step size shrinks as t grows -> local refinement late
                r2 = np.random.rand(self.dim)
                step = (low + r2 * (high - low)) / t
                candidate_p2 = ospreys[i] + step
                candidate_p2 = np.clip(candidate_p2, low, high)

                score_p2 = self.fitness_fn(candidate_p2)
                if score_p2 > fitness[i]:
                    ospreys[i] = candidate_p2
                    fitness[i] = score_p2

            # Update global best after full population sweep
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_score:
                best_score = fitness[gen_best_idx]
                best_osprey = ospreys[gen_best_idx].copy()

            history.append(best_score)

        return best_osprey, best_score, history
