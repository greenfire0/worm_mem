"""
main_pure_nomad.py – Pure‑NOMAD on CPU‑only JAX
"""

# 1 ─ Imports
import time, numpy as np, jax, jax.numpy as jnp
from Algorithms.Pure_NOMAD_EDIT import PureNOMAD
from Algorithms.algo_utils       import evaluate_fitness_static
from Worm_Env.celegan_env        import WormSimulationEnv
from Worm_Env.connectome         import WormConnectome

# 2 ─ Baseline weights & env
weights = jnp.asarray(WormConnectome().weights, dtype=jnp.float32)

# 3 ─ Python fitness
def python_fitness(vec:WormConnectome,env=None, prob=[0], interval=250, episodes=1):
    return evaluate_fitness_static(
        WormConnectome(init_weights=vec.weights),
        WormSimulationEnv(graphing=False), prob, interval, episodes,
    )

def fitness_batch(pop_jax):
    return jnp.asarray([python_fitness(w) for w in pop_jax])

# 4 ─ Optimiser & STATE
algo:PureNOMAD   = PureNOMAD(population_size=1, solution=weights, subset_size=49,eval_function= python_fitness)
params = algo.default_params.replace(step_size=0.5, crossover_rate=0.1)

key   = jax.random.PRNGKey(0)
state = algo._init(key, params)          # ← always returns a State object
# 5 ─ Loop
for g in range(14):
    key, k1, k2 = jax.random.split(key, 3)

    # ask  → 3 outputs
    pop, state = algo._ask(k1, state, params)  #ask will also give fitness
    # evaluate population
    fit   = fitness_batch(pop) ##

    # tell  → state only
    

    state = algo._tell(k2, pop, fit, state, params)

    print(f"Gen {g:02d} | best fitness = {float(jnp.min(fit)):.3f}")


# 6 ─ Save best
np.save("best_connectome.npy", np.asarray(algo.get_best_solution(state)))
print("Saved best_connectome.npy")
