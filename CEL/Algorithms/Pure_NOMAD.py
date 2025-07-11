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
    def __init__(self, population_size:int = 14,pattern:List= [0,1],  total_episodes:int=1, training_interval:int=250,matrix_shape:int= 3951):
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
                            futures.append(evaluate_fitness_nomad.remote(
                                evaluate_fitness_static,
                                candidate_weights= candidate.weights,
                                env = WormSimulationEnv(), 
                                prob_type = self.probs,
                                interval = self.training_interval,
                                episodes = self.total_episodes,
                                ind=np.random.choice(self.matrix_shape, size=49, replace=False),
                                bounds = 4,
                                bb_eval = 200,
                                verify = False # turn this on for debugging
                            ))

                results = ray.get(futures)
                for a,(optimized_weights,fitness) in enumerate(results):
                        self.population[a].weights = optimized_weights ## this incorrectly sets the number of weights and it ends up bieng less than the correct amount its somehting wrong with the  way weights are bieng set
                        if (len(self.population[a].weights)) !=len(optimized_weights):
                            ind = np.where(optimized_weights!= self.population[a+1].weights)
                            print(len(self.population[a].weights))
                        fitnesses.append(fitness)
                del results
                best_index = np.argmax(fitnesses)
                best_fitness = fitnesses[best_index]
                best_weights = (self.population[best_index].weights)
                best_worm = WormConnectome(init_weights=best_weights)
                check = evaluate_fitness_static(best_weights,WormSimulationEnv(),self.probs,self.training_interval,self.total_episodes)

                assert best_fitness ==  check
                print(f"Generation {generation + 1} best fitness: {best_fitness}")
                
                self.population,fitnesses = select_parents(self.population,fitnesses, self.population_size // 2 )
                self.population.extend(crossover(self.population, fitnesses, self.population_size - len(self.population),self.matrix_shape))
                self.population.append(best_worm)
                
                write_worm_to_csv(filename, best_worm,max_rows=generations)
            #return best_weights
        finally:
            ray.shutdown()