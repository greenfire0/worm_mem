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
    def __init__(self, population_size:int = 1,pattern:List= [0],  total_episodes:int=1, training_interval:int=250,matrix_shape:int= 3951):
        self.population_size:int = population_size
        self.matrix_shape:int = matrix_shape
        self.total_episodes:int = total_episodes
        self.training_interval:int = training_interval
        self.probs:List[int] = pattern
        self.population = initialize_population(self.population_size)

    def run(self, generations=50, batch_size=32,filename:str = "Pure_nomad"):
        try:
            for generation in tqdm(range(generations), desc="Generations"):
                population_batches = [self.population[i:i+batch_size] for i in range(0, len(self.population), batch_size)]
                fitnesses,futures = [],[]
                for batch in population_batches:
                    for candidate in (batch):
                            futures.append(evaluate_fitness_ray.remote(
                                candidate_weights= candidate.weights,
                                env = WormSimulationEnv(), 
                                prob_type = self.probs,
                                interval = self.training_interval,
                                episodes = self.total_episodes
                            ))     


                results = ray.get(futures)
                for a,result in enumerate(results):
                        
                        new_vals = np.asarray(result[0][1], dtype=np.float64)   # ← one line
                        diff     = self.population[a].weights[result[0][0]] != new_vals
                        print(new_vals[diff],self.population[a].weights[result[0][0]][diff])
                        fitnesses.append(np.max([(result[1]),0]))
                
                del results
                best_index = np.argmax(fitnesses)
                best_fitness = fitnesses[best_index]
                best_weights = (self.population[best_index].weights)
                best_worm = WormConnectome(init_weights=best_weights)
                
                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                """
                self.population,fitnesses = select_parents(self.population,fitnesses, self.population_size // 2 )
                self.population.extend(crossover(self.population, fitnesses, self.population_size - len(self.population),self.matrix_shape))
                self.population.append(best_worm)
                
                write_worm_to_csv(filename, best_worm,max_rows=generations)"""

            #return best_weights
        finally:
            ray.shutdown()