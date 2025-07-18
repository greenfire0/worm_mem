# -*- coding: utf-8 -*-
"""pure_nomad_max.py
========================================
A reward‑maximising, **PyNomad‑backed** evolutionary optimiser that keeps the
EvoSax API (ask / tell) but replaces the internal JAX "poll" with a true NOMAD
local search on a *subset* of weights for every elite.

Key points
----------
* **Maximises** episodic reward ⇒ higher is better.
* Uses the remote ``evaluate_fitness_nomad`` helper (Ray + PyNomad) to perform
  the actual local refinement.
* Falls back to the original random NOMAD‑style step if the remote optimiser
  is not configured.

The class can be slotted into any EvoSax training loop exactly like the
original ``PureNOMAD`` – just swap the import.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from optax import constant_schedule

import ray  # ensure ray.init() has been called in your driver script

from evosax.algorithms.population_based.base import (
    PopulationBasedAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn,
)
from evosax.types import Fitness, Population, Solution
from evosax.core.fitness_shaping import identity_fitness_shaping_fn

# ─────────────────────────────────────────────────────────────────────────────
# Remote NOMAD evaluator ------------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
from Algorithms.algo_utils import evaluate_fitness_nomad, evaluate_fitness_static  # adjust import as needed


# ─────────────────────────────────────────────────────────────────────────────
# 1 │ Parameter & state dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@struct.dataclass
class Params(BaseParams):
    crossover_rate: float = 0.1
    step_size: float = 0.5               # not used by PyNomad but kept for API
    step_coeff_down: float = 0.5         # idem
    step_coeff_up: float = 1.2


@struct.dataclass
class State(BaseState):
    population: Population               # (μ, D)
    fitness:    Fitness                  # (μ,)
    step_size:  jax.Array                # scalar σ (kept for compat)


# ─────────────────────────────────────────────────────────────────────────────
# 2 │ Pure‑NOMAD‑Max optimiser
# ─────────────────────────────────────────────────────────────────────────────

class PureNOMAD(PopulationBasedAlgorithm):
    """Evolutionary optimiser that *maximises* reward and calls PyNomad
    for a true local search around each elite. Everything remains pure‑Python
    outside JAX jit scopes, so you can still jit the *fitness* if desired.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        eval_function:Callable,
        *,
        subset_size: int = 49,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
        std_schedule: Callable = constant_schedule(1.0),
        # extra args for the remote NOMAD helper ----------------------------
        ray_pool: Optional[ray.util.ActorPool] = None,
        env=None,
        prob_type: List[int] | None = None,
        interval: int = 250,
        episodes: int = 1,
        bounds: float = 0.1,
        bb_eval: int = 200,
        init_noise: bool = False,
    ) -> None:
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)
        self.std_schedule = std_schedule
        self.elite_ratio = 0.5
        self._subset_size = int(subset_size)

        # Store NOMAD context -------------------------------------------------
        self._ray_pool   = ray_pool
        self._env        = env
        self._prob_type  = prob_type if prob_type is not None else [0]
        self._interval   = interval
        self._episodes   = episodes
        self._bounds     = bounds
        self._bb_eval    = bb_eval
        self._init_noise = init_noise
        self._eval_func  = eval_function 

    # ────────────────────────────────────────────────────────────────────
    # API helpers
    # ────────────────────────────────────────────────────────────────────

    @property
    def _default_params(self) -> Params:  # noqa: D401
        return Params()

    def init(self, key: jax.Array, params: Optional[Params] = None) -> State:  # type: ignore[override]
        if params is None:
            params = self._default_params
        return self._init(key, params)

    def _nomad_poll(self, x_numpy: np.ndarray) -> np.ndarray:
        """Call the Ray + PyNomad helper to locally maximise reward."""
        #if evaluate_fitness_nomad is None or self._env is None:
            # graceful fallback – random step
        #    return x_numpy

        ind = np.random.choice(self.num_dims, self._subset_size, replace=False)
        fut = evaluate_fitness_nomad.remote(
            self._eval_func,         
            x_numpy,
            self._env,
            self._prob_type,
            self._interval,
            self._episodes,
            ind,
            self._bounds,
            self._bb_eval,
        )
        new_cand, _best_reward = ray.get(fut)
        #print(_best_reward)
        return np.asarray(new_cand.weights, dtype=np.float32)


    # ────────────────────────────────────────────────────────────────────
    # _init / _ask / _tell
    # ────────────────────────────────────────────────────────────────────

    def _init(self, key: jax.Array, params: Params) -> State:  # noqa: D401
        """Create generation 0.

        If `self._init_noise` is True, every parameter vector is the
        base `solution` plus N(0, σ²) noise, where σ comes from
        `self.std_schedule(0)`.  Otherwise, the whole population is an
        exact copy of the base solution.
        """
        pop_key, _ = jax.random.split(key)

        # (μ, D) array of the base solution
        base = jnp.broadcast_to(
            jnp.asarray(self.solution, jnp.float32),
            (self.population_size, self.num_dims),
        )

        if self._init_noise:
            sigma = self.std_schedule(0)  # scalar
            noise = jax.random.normal(
                pop_key, (self.population_size, self.num_dims), dtype=jnp.float32
            ) * sigma
            population = base + noise
        else:
            population = base  # no perturbation

        return State(
            population=population,
            fitness=jnp.full((self.population_size,), -jnp.inf),  # maximise!
            step_size=jnp.asarray(params.step_size, jnp.float32),
            best_solution=jnp.asarray(self.solution, jnp.float32),
            best_fitness=-jnp.inf,
            generation_counter=0,
        )

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> Tuple[Population, State]:
        """Generate the next population.

        For every elite we perform a **true NOMAD poll** via PyNomad; the rest
        of the population is filled with crossover offspring.
        """
        k_elite, k_rand, k_cross = jax.random.split(key, 3)

        # Elites – pick the top μ/2 by *highest* fitness -----------------
        idx = jnp.argsort(state.fitness)[::-1]
        elites = state.population[idx][: self.num_elites]

        # Convert elites to numpy and run local poll ---------------------
        elites_np = np.asarray(elites, dtype=np.float32)
        refined = [self._nomad_poll(e) for e in elites_np]
        refined_np = np.stack(refined, axis=0).astype(np.float32)
        refined_jax = jnp.asarray(refined_np)

        # Crossover offspring -------------------------------------------
        off_count = self.population_size - refined_jax.shape[0]
        if off_count > 0:
            cross_keys = jax.random.split(k_cross, off_count)
            p1_idx = jax.random.randint(cross_keys[0], (off_count,), 0, refined_jax.shape[0])
            p2_idx = jax.random.randint(cross_keys[1], (off_count,), 0, refined_jax.shape[0])
            p1, p2 = refined_jax[p1_idx], refined_jax[p2_idx]

            def _cx(k, a, b):
                mask = jax.random.uniform(k, a.shape) < params.crossover_rate
                return a * (1.0 - mask) + b * mask

            offspring = jax.vmap(_cx)(cross_keys, p1, p2)
            population_next = jnp.concatenate([refined_jax, offspring])
        else:
            population_next = refined_jax

        return population_next[: self.population_size], state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        """Update state – *maximise* fitness."""
        improved = jnp.max(fitness) > jnp.max(state.fitness)
        step = jnp.where(improved, params.step_coeff_down, params.step_coeff_up)
        return state.replace(
            population=population,
            fitness=fitness,
            step_size=state.step_size * step,
        )
