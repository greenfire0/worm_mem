# -*- coding: utf-8 -*-
"""pure_nomad_generic.py
====================================================
A **general‑purpose evolutionary optimiser** that couples global search
(crossover + mutation) with local refinement via the NOMAD derivative‑free
solver (through *PyNomad*). It is *not* tied to any specific domain – the user
supplies an ``objective_fn`` that maps a NumPy 1‑D parameter vector → scalar
fitness.  Higher fitness means *better*.  The optimiser handles everything
else.

Example
-------
```python
import numpy as np
from pure_nomad_generic import PureNOMAD

def sphere(x: np.ndarray) -> float:
    return -np.sum(x ** 2)         # maximise ⇒ minimise (–sphere)

opt = PureNOMAD(
    population_size=64,
    dimension=30,
    objective_fn=sphere,
    subset_size=10,
    bounds=0.2,
    max_bb_eval=250,
)

best_x, best_fit = opt.run(generations=200)
print(best_fit, best_x)
```

Dependencies: ``numpy``, ``PyNomad>=0.9`` (optional ``ray`` for parallel NOMAD
calls; falls back to serial execution if not available).
"""
from __future__ import annotations

import math
import numpy as np
from tqdm import tqdm                      
import numpy.typing as npt
from typing import Callable, List, Tuple, Sequence, Optional

try:
    import ray  # type: ignore
    _RAY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RAY_AVAILABLE = False

import PyNomad  # type: ignore

__all__ = [
    "PureNOMAD",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _uniform_crossover(parents: npt.NDArray, probs: npt.NDArray) -> npt.NDArray:
    """Vectorised uniform crossover.

    Parameters
    ----------
    parents : (P, D) parent weight matrix
    probs   : (P,) selection probabilities ∝ fitness (must sum to 1)

    Returns
    -------
    offspring : (P, D)  new individuals (same count as parents)
    """
    P, D = parents.shape
    p1_idx = np.random.choice(P, size=P, p=probs)
    p2_idx = np.random.choice(P, size=P, p=probs)

    p1, p2 = parents[p1_idx], parents[p2_idx]
    mask = np.random.rand(P, D) < 0.5  # 50‑50 gene swap
    return np.where(mask, p1, p2)


def _random_reset_mutation(pop: npt.NDArray, n_coords: int, low: float, high: float) -> None:
    """In‑place random‑reset mutation on *n_coords* indices per individual."""
    N, D = pop.shape
    for i in range(N):
        idx = np.random.choice(D, n_coords, replace=False)
        pop[i, idx] = np.random.uniform(low, high, size=n_coords)


# ─────────────────────────────────────────────────────────────────────────────
# NOMAD wrapper (minimises –fitness)
# ─────────────────────────────────────────────────────────────────────────────

def _nomad_local_search(
    fitness_fn: Callable[[npt.NDArray], float],
    x0: npt.NDArray,                    # starting point (sub‑vector)
    full_x: npt.NDArray,                # full vector (will be copied)
    ind: npt.NDArray[np.intp],          # indices we optimise (1‑D)
    bounds: float,
    max_bb_eval: int,
) -> Tuple[npt.NDArray, float]:
    """Run NOMAD on the coordinate slice ``ind``.

    Returns the updated *full* vector and its fitness.
    """
    lb = (x0 - bounds).tolist()
    ub = (x0 + bounds).tolist()

    def obj(eval_point):
        candidate_edit = []
        vect = full_x.copy()
        for a in range(len(ind)):
                candidate_edit.append(eval_point.get_coord(a))
        vect[ind] = candidate_edit
        
        eval_value = -1*fitness_fn(vect)
        return eval_value

    opts = [
        "DISPLAY_DEGREE 0",
        "DISPLAY_STATS BBE OBJ",
        f"MAX_BB_EVAL {max_bb_eval}",
    ]
    res = PyNomad.optimize(obj, x0.tolist(), lb, ub, opts)
    #if "Problem with starting point evaluation" in res['stop_reason']:
    #    raise ValueError('Problem with starting point evaluation, no more points to evaluate',f'bound {bounds}')
    best_slice = np.asarray(res["x_best"], dtype=np.float64)
    if best_slice.size != ind.size:      # NOMAD didn’t return a point
        best_slice = x0                  # keep the original slice
        best_fit   = fitness_fn(full_x)  # evaluate it
        full_out = full_x.copy()
    else:
        best_fit   = -res["f_best"]
        full_out = full_x.copy()
        full_out[ind] = best_slice

    return full_out, best_fit


if _RAY_AVAILABLE:
    # Ray remote wrapper -----------------------------------------------------
    @ray.remote(num_cpus=1)
    def _nomad_remote(fn, x0, full_x, ind, bounds, max_bb_eval):  # type: ignore[valid-type]
        return _nomad_local_search(fn, x0, full_x, ind, bounds, max_bb_eval)

# ─────────────────────────────────────────────────────────────────────────────
# Main optimiser class
# ─────────────────────────────────────────────────────────────────────────────

class PureNOMAD:
    """Composable evolutionary optimiser with NOMAD local search.

    Parameters
    ----------
    population_size   : number of individuals (μ)
    dimension         : parameter vector length  *(ignored if `init_pop` given)*
    objective_fn      : callable `f(x) -> float`, *to be maximised*
    subset_size       : number of coordinates refined by NOMAD per elite
    bounds            : ±box around the elite weight slice passed to NOMAD
    max_bb_eval       : NOMAD budget per call
    n_elites          : how many top individuals get local optimisation
    n_mutate_coords   : coordinates mutated per individual (random reset)
    crossover_rate    : fraction of new population produced via crossover
    init_pop          : optional initial population array (μ, D)
    low, high         : range for random initialisation & mutation
    use_ray           : parallelise NOMAD calls with Ray if available
    seed              : RNG seed (int or None)
    """

    def __init__(
        self,
        population_size: int,
        *,
        dimension: int | None = None,
        objective_fn: Callable[[npt.NDArray], float],
        subset_size: int = 20,
        bounds: float = 0.1,
        max_bb_eval: int = 200,
        n_elites: int | None = None,
        n_mutate_coords: int = 5,
        crossover_rate: float = 0.5,
        init_pop: npt.NDArray | None = None,
        init_vec: npt.NDArray | None = None,
        low: float = -1.0,
        high: float = 1.0,
        use_ray: bool | None = None,
        seed: int | None = None,
    ) -> None:
        if init_pop is None and dimension is None:
            raise ValueError("Either 'dimension' or 'init_pop' must be provided.")

        if init_pop.ndim < 2:
            raise ValueError("init_pop should be at least 2 dimensions"\
                              f"(given: {init_pop.ndim})") 
        if init_pop and init_vec:
            raise ValueError("Both")
        self.D = dimension if init_pop is None else init_pop.shape[1]
        self.rng = np.random.default_rng(seed)

        self.mu = population_size
        # population ----------------------------------------------------
        if init_pop is not None:
            if init_pop.shape != (population_size, self.D):
                raise ValueError("init_pop must have shape (population_size, dimension)")
            self.pop: npt.NDArray = init_pop.copy()
        else:
            self.pop = self.rng.uniform(low, high, size=(population_size, self.D))

        self.obj = objective_fn
        self.subset_size = subset_size
        self.bounds = bounds
        self.max_bb_eval = max_bb_eval
        self.n_elites = n_elites if n_elites is not None else population_size // 2
        self.n_mutate_coords = n_mutate_coords
        self.crossover_rate = crossover_rate
        self.low, self.high = low, high
        self.use_ray = _RAY_AVAILABLE if use_ray is None else use_ray and _RAY_AVAILABLE

        if self.use_ray and not ray.is_initialized():  # pragma: no cover
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        # internal bookkeeping -----------------------------------------
        self.generation = 0
        self._best_fit = -math.inf
        self._best_x: Optional[npt.NDArray] = None

    # ────────────────────────────────────────────────────────────────
    # Core loop
    # ────────────────────────────────────────────────────────────────

    def _evaluate_population(self) -> npt.NDArray[np.float64]:
        """Vectorised objective call; returns (μ,) fitness array."""
        return np.asarray([self.obj(ind) for ind in self.pop], dtype=np.float64)

    def _select_parents(self, fits: npt.NDArray[np.float64]) -> Tuple[npt.NDArray, npt.NDArray]:
        idx = np.argsort(fits)[-self.mu // 2:]  # top‑half
        return self.pop[idx], fits[idx]

    def _make_offspring(self, parents: npt.NDArray, parent_fits: npt.NDArray) -> npt.NDArray:
        probs = parent_fits / parent_fits.sum()
        offspring = _uniform_crossover(parents, probs)
        _random_reset_mutation(offspring, self.n_mutate_coords, self.low, self.high)
        return offspring

    def _local_search_elites(self, fits: npt.NDArray) -> None:
        elite_idx = np.argsort(fits)[-self.n_elites:]
        tasks = []
        for idx in elite_idx:
            indiv = self.pop[idx]
            slice_idx = self.rng.choice(self.D, self.subset_size, replace=False)
            x0 = indiv[slice_idx].copy()
            if self.use_ray:
                tasks.append(_nomad_remote.remote(
                    self.obj, x0, indiv, slice_idx, self.bounds, self.max_bb_eval
                ))
            else:
                tasks.append(_nomad_local_search(self.obj, x0, indiv, slice_idx,
                                                 self.bounds, self.max_bb_eval))
        results = ray.get(tasks) if self.use_ray else tasks
        for new_x, new_fit in results:
            # replace if improved ----------------------------------
            if new_fit > fits.mean():
                worst = np.argmin(fits)
                self.pop[worst] = new_x
                fits[worst] = new_fit

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def step(self) -> Tuple[float, npt.NDArray]:
        """Run one generation; return best fitness and best vector."""
        fits = self._evaluate_population()

        # book‑keep global best ---------------------------------------
        best_idx = np.argmax(fits)
        if fits[best_idx] > self._best_fit:
            self._best_fit = float(fits[best_idx])
            self._best_x   = self.pop[best_idx].copy()

        # local NOMAD refinement on elites ----------------------------
        self._local_search_elites(fits)

        # produce next generation ------------------------------------
        parents, parent_fits = self._select_parents(fits)
        offspring = self._make_offspring(parents, parent_fits)
        self.pop = np.vstack([parents, offspring])[: self.mu]

        self.generation += 1
        return self._best_fit, self._best_x.copy()

    def run(self, generations: int) -> Tuple[npt.NDArray, float]:
        """Run optimisation for `generations` steps. Return best_x, best_fit."""
        pbar = tqdm(range(generations), desc="Generations", unit="gen")
        for _ in pbar:
            self.step()
            # update «postfix» field (appears on the right)
            pbar.set_postfix(best_fit=f"{self._best_fit: .4f}")

        pbar.close()
        return self._best_x.copy(), self._best_fit

    # string representation -----------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PureNOMAD(mu={self.mu}, D={self.D}, gen={self.generation}, "
            f"best={self._best_fit:.3e})"
        )
