from Worm_Env.connectome import WormConnectome
from Worm_Env.celegan_env import WormSimulationEnv
import numpy as np
import ray
import PyNomad
import numpy.typing as npt
import time
from numba import njit
from typing import List
@staticmethod
def initialize_population(population_size:int):
    population = []
    for _ in range(population_size):
        population.append(WormConnectome())
    return population


@staticmethod
def select_parents(population, fitnesses, num_parents):
    """
    Returns
    -------
    parents           : list[WormConnectome]
        The `num_parents` individuals with highest fitness.
    parent_fitnesses  : list[float]
        Their corresponding fitness values (same order).
    """
    idx = np.asarray(fitnesses).argsort()[-num_parents:]  # top-k indices
    selected_pop   = [population[i] for i in idx]
    selected_fits  = [fitnesses[i]  for i in idx]
    return selected_pop, selected_fits

@staticmethod
def crossover(parents, fitnesses, num_offspring, matrix_shape):
    """Return `num_offspring` new WormConnectome objects (vectorised)."""

    # ------------------------------------------------------------------
    # 0.  Pre-compute reusable arrays once
    # ------------------------------------------------------------------
    par_count   = len(parents)
    fitness_arr = np.asarray(fitnesses, dtype=np.float64)

    # guard against zero-sum
    if fitness_arr.sum() == 0:
        fitness_probs = np.full(par_count, 1.0 / par_count)
    else:
        fitness_probs = fitness_arr / fitness_arr.sum()

    # stack weights into a 2-D view   shape: (P, W)
    weights_2d = np.vstack([p.weights for p in parents])
    if len(parents) != len(fitnesses):
        print("❌  len(population) =", len(parents),
            "len(fitnesses) =", len(fitnesses))
        raise RuntimeError("Population/fitness length mismatch")
    # ------------------------------------------------------------------
    # 1.  Sample parent indices in one shot
    # ------------------------------------------------------------------
    p1_idx = np.random.choice(par_count,
                              size=num_offspring,
                              p=fitness_probs)
    p2_idx = np.random.choice(par_count,
                              size=num_offspring,
                              p=fitness_probs)

    # ------------------------------------------------------------------
    # 2.  Compute crossover probabilities for every offspring
    #     pcross = (fit1 / (fit1 + fit2)) ** 1.2
    # ------------------------------------------------------------------
    fit1 = fitness_probs[p1_idx]
    fit2 = fitness_probs[p2_idx]
    pcross = (fit1 / (fit1 + fit2)) ** 1.2      #  shape: (O,)

    # ------------------------------------------------------------------
    # 3.  Build offspring weight matrix in one vectorised shot
    # ------------------------------------------------------------------
    # random matrix  (O, W)
    rnd = np.random.rand(num_offspring, matrix_shape)
    mask = rnd < pcross[:, None]                # broadcast

    offspring_weights = np.where(mask,
                                 weights_2d[p1_idx],
                                 weights_2d[p2_idx])   # shape (O, W)

    # ------------------------------------------------------------------
    # 4.  Wrap into WormConnectome objects
    # ------------------------------------------------------------------
    return [WormConnectome(init_weights=w) for w in offspring_weights]

@ray.remote
def evaluate_fitness_ray(candidate_weights, env, prob_type,interval,episodes):
    sum_rewards = 0
    candidate = WormConnectome(init_weights=candidate_weights)
    for a in prob_type:
        env.reset(a)
        for _ in range(episodes):  # total_episodes
            observation = env._get_observations()
            for s in range(interval):  # training_interval
                movement = candidate.move(observation[0], observation[4])
                next_observation, reward = env.step(movement, s)
                observation = next_observation
                sum_rewards+=reward
    return sum_rewards

@staticmethod
def evaluate_fitness_static(candidate_weights, env:WormSimulationEnv, prob_type ,interval,episodes):
        sum_rewards = 0
        candidate = WormConnectome(init_weights=candidate_weights)
        for a in prob_type:
            env.reset(a)
            for _ in range(episodes):  # total_episodes
                observation = env._get_observations()
                for s in range(interval):  # training_interval
                    movement = candidate.move(observation[0], observation[4])
                    next_observation, reward = env.step(movement, s)
                    observation = next_observation
                    sum_rewards+=reward
        return sum_rewards



@staticmethod
def mutate(offspring,matrix_shape, n=5):
    for child in offspring:
                indices_to_mutate = np.random.choice(matrix_shape, size=n, replace=False)
                new_values = np.random.uniform(low=-20, high=20, size=n)
                child.weights[indices_to_mutate] = new_values
    return offspring


@ray.remote(max_retries=0) # type: ignore[arg-type]
def evaluate_fitness_nomad(func, candidate_weights:npt.NDArray[np.float64], env, prob_type, interval, episodes,ind,bounds:int,bb_eval,verify:bool=False):
        wall_start = time.perf_counter() 
        cpu_start  = time.process_time()
        if ind.size == 0:
                raise ValueError(f"Please pass indicies,{ind}")
        x0:npt.NDArray[np.float64] = np.array(candidate_weights[ind])
        x0 = np.array(candidate_weights[ind])
        lower_bounds = (x0 - bounds).tolist()
        upper_bounds = (x0 + bounds).tolist()
        x0 = x0.tolist()
        
        params = [
            'DISPLAY_DEGREE 0', 
            'DISPLAY_STATS BBE BLK_SIZE OBJ', 
            'BB_MAX_BLOCK_SIZE 4',
            f'MAX_BB_EVAL {bb_eval}'
        ]
        wrapper = BlackboxWrapper(func,env, prob_type, interval, episodes,ind,candidate_weights)
        result = PyNomad.optimize(wrapper.blackbox_block, x0, lower_bounds, upper_bounds,params)
        # Use NOMAD's minimize function with blackbox_block and pass additional args

        if verify:
            w_test = np.copy(candidate_weights)
            w_test.setflags(write=True)        
            w_test[ind] = np.copy(result['x_best'])
            fitness_verify = func(
                                        w_test,
                                        env,
                                        prob_type,
                                        interval,
                                        episodes)
            #print("fitness",-result['f_best'],"fitness",fitness_verify)
            assert abs(fitness_verify+result['f_best'])<0.1,( w_test[ind]==result['x_best'], "\nResults\n",fitness_verify,result['f_best'])
            del w_test,fitness_verify
        
        cpu_total  = time.process_time() - cpu_start
        wall_total = time.perf_counter() - wall_start
        cpu_fit    = wrapper.time                # only fitness calls
        cpu_nomad  = max(cpu_total - cpu_fit, 0) # PyNomad + copy + misc
        del wrapper
        print(result)
        return ([ind,result['x_best']],-result['f_best'],cpu_fit,wall_total,cpu_nomad)



class BlackboxWrapper:
    def __init__(self, func, env, prob_type, interval, episodes,index,cand):
        self.env = env
        self.func = func
        self.prob_type = prob_type
        self.interval = interval
        self.episodes = episodes
        self.ind = index
        self.candidate = cand
        self.time =0.0

    def blackbox(self, eval_point):
            self.candidate_edit = []
            self.candidate_weights = np.copy(self.candidate).astype(np.float64)
            for a in range(len(self.ind)):
                self.candidate_edit.append(eval_point.get_coord(a))
            self.candidate_weights[self.ind] = self.candidate_edit
            tic = time.process_time()
            eval_value = -1*self.func(
                    self.candidate_weights, self.env, self.prob_type, 
                    self.interval, self.episodes)
            self.time += time.process_time() - tic
            eval_point.setBBO(str(eval_value).encode('utf-8'))
            del self.candidate_weights
            return True

    def blackbox_block(self, eval_block):
        eval_state = []
        for index in range(eval_block.size()):
            eval_point = eval_block.get_x(index)
            eval_state.append(self.blackbox(eval_point))
        return eval_state


@staticmethod
@njit
def lasso_reg(candidate_weights, original, lambda_=0.1):
        num_differences = np.count_nonzero(candidate_weights != original)
        penalty = -lambda_ * np.power(num_differences, 1.3)
        return penalty