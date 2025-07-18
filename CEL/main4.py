from algorithms.Pure_NOMAD import PureNOMAD
import numpy as np, math

def rastrigin(x: np.ndarray) -> float:
    return -(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * math.pi * x)))


d_total = 3000
d_good  = 2000          # 2/3 correct
d_bad   = d_total - d_good

# build x
x = np.zeros(d_total, dtype=float)
x[d_good:] = np.random.uniform(-5.12, 5.12, size=d_bad)

opt = PureNOMAD(
    population_size = 2048,        # larger μ gives coverage in 3000‑D
    dimension       = 3000,
    objective_fn    = rastrigin,
    subset_size     = 200,         # NOMAD refines 200 coords per elite
    bounds          = 2.0,         # ±2 search box (start coarse)
    max_bb_eval     = 400,         # per‑call budget
    n_mutate_coords = 30,          # small random resets each gen
    init_pop=x
)

best_x, best_f = opt.run(generations = 1000)
print("best fitness:", best_f)