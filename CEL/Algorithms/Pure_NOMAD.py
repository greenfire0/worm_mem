import numpy as np
import ray
from Worm_Env.connectome import WormConnectome
from Worm_Env.celegan_env import WormSimulationEnv
from Worm_Env.weight_dict import muscles,muscleList,mLeft,mRight
from tqdm import tqdm
import csv
from Algorithms.algo_utils import initialize_population, select_parents,\
crossover, evaluate_fitness_ray,evaluate_fitness_static,BlackboxWrapper,evaluate_fitness_nomad
from util.snip import write_worm_to_csv
import numpy.typing as npt
from typing import List

class Pure_NOMAD:
    def __init__(self, population_size:int = 64,pattern:List= [0],  total_episodes:int=1, training_interval:int=100,matrix_shape:int= 3951):
        self.population_size:int = population_size
        self.matrix_shape:int = matrix_shape
        self.total_episodes:int = total_episodes
        self.training_interval:int = training_interval
        self.probs:List[int] = pattern
        self.population = initialize_population(self.population_size)

    def run(self, generations=3, batch_size=32,filename:str = "Pure_nomad"):
        try:
            for generation in tqdm(range(generations), desc="Generations"):
                population_batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
                fitnesses,futures = [],[]
                for batch in population_batches:
                    for candidate in (batch):
                            futures.append(evaluate_fitness_nomad.remote(
                                evaluate_fitness_static,
                                candidate.weights,
                                env = WormSimulationEnv(), 
                                prob_type = self.probs,
                                interval = self.training_interval,
                                episodes = self.total_episodes,
                                ind=np.random.choice(self.matrix_shape, size=49, replace=False),
                                bounds = 4,
                                bb_eval = 250,
                                verify = False # turn this on for debugging
                            ))

                results = ray.get(futures)
                for a,(optimized_worm,fitness) in enumerate(results):
                        self.population[a] = optimized_worm 
                        fitnesses.append(fitness)
                del results
                best_index = np.argmax(fitnesses)
                best_fitness = fitnesses[best_index]
                best_worm = (self.population[best_index])

                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                
                self.population,fitnesses = select_parents(self.population,fitnesses, self.population_size // 2 )
                self.population.extend(crossover(self.population, fitnesses, self.population_size - len(self.population),self.matrix_shape))
                self.population.append(best_worm)
                
                write_worm_to_csv(filename, best_worm,max_rows=generations)
            #return best_weights
        finally:
            ray.shutdown()