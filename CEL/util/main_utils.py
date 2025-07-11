import numpy as np
from typing import Dict
from Worm_Env.celegan_env import WormSimulationEnv
import ray
import multiprocessing
# Genetic Algorithm Variants
from Algorithms.Evolutionary_Algorithm import Genetic_Dyn_Algorithm as GD_EA
from Algorithms.Hybrid_NOMAD import Genetic_Dyn_Algorithm as GD_EA_Nomad
from Algorithms.Pure_NOMAD import Genetic_Dyn_Algorithm as GD_PureNomad
from Algorithms.RandParam_50_NOMAD import Genetic_Dyn_Algorithm as GD_RandomNomad
from Algorithms.EVO_NOMAD import Genetic_Dyn_Algorithm as GD_EVO
from Algorithms.OaI_es import train_openai_es
from Algorithms.CMA_ES import train_cma_es
# Graphs
from graphs.Graph_pos_over_time import Genetic_Dyn_Algorithm as GD_Pos
from graphs.Graph_fitness_over_time import Genetic_Dyn_Algorithm as GD_Graph
from graphs.Graph_path_over_gen import Genetic_Dyn_Algorithm as GD_PathGen
from graphs.wpi import search_connection_impacts, calc_simular
from graphs.graphing import graph_results, graph_trained_worms, graph_agg
from graphs.Graph_fitness_over_time_old import Genetic_Dyn_Algorithm as GD_Graph_Old
from CEL.graphs.graph_video import Genetic_Dyn_Video
import numpy.typing as npt
from util.write_read_txt import delete_arrays_csv_if_exists



def select_ga_class(config:Dict):
    """
    Selects which Genetic_Dyn_Algorithm variant to use based on the config.
    Now the dictionary keys are more descriptive.
    """
    variant_map = {
        "EVO_NOMAD":                      GD_EVO,
        "graph_positions_over_time":      GD_Pos,      # old "pos"
        "graph_path_quartile_evolution":  GD_PathGen,  # new "path"
        "graph_fitness_over_time":        GD_Graph,    # old "graph"
        "graph_fitness_over_time_legacy": GD_Graph_Old,# old "graph_old"
        "pure_nomad_algorithm":           GD_PureNomad,# old "pure_nomad"
        "random_nomad_algorithm":         GD_RandomNomad,  # old "random_nomad"
        "standard_evolutionary_algorithm":GD_EA,       # old "ea"
        "nomad_evolutionary_algorithm":   GD_EA_Nomad, # old "ea_nomad"
    }
    return variant_map.get(config["ga_variant"], GD_PureNomad)


def calculate_worm_suffering_index(config:Dict,values_list:npt.NDArray[np.float64],length:int):
    """Calculates a 'worm suffering index' by searching for connection impacts, etc."""
    env = WormSimulationEnv()
    ci_3 = search_connection_impacts(
        original_genome=[],
        matrix_shape=0,
        env=env,
        prob_type=[3],
        interval=config["training_interval"],
        episodes=config["total_episodes"]
    )
    ci_5 = search_connection_impacts(
        original_genome=[],
        matrix_shape=0,
        env=env,
        prob_type=[5],
        interval=config["training_interval"],
        episodes=config["total_episodes"]
    )
    #ci_diff = ci_3 - ci_5 was used in the past for something now not used
    calc_simular(ci_3, ci_5)


def test_last_generations(config:Dict):
    """Tests and prints the last generations from arrays.csv (if you store them)."""
    env = WormSimulationEnv()
    ga_instance = GD_Graph(
        population_size=config["population_size"],
        pattern=config["food_patterns"],
        total_episodes=config["total_episodes"],
        training_interval=config["training_interval"]
    )
    ga_instance.run_and_print_last_generations(env, '24hr')


def graph_training_results(config:Dict,values_list:npt.NDArray[np.float64]):
    """Graphs results of a single worm's training."""
    csv_name = "5"
    graph_results(config["path"], csv_name, values_list)


def graph_trained_population(config:Dict,values_list:npt.NDArray[np.float64]):
    """Graphs the results of trained worms at the end of training."""
    graph_trained_worms(base_path=config["path"], values_list=values_list)


def graph_aggregates(config:Dict,values_list:npt.NDArray[np.float64]):
    """Graphs aggregate results from multiple runs."""
    graph_agg(base_path=config["path"], values_list=values_list)


def clean_environment():
    """Cleans the environment by deleting arrays.csv if it exists."""
    print("Clearing Environment...")
    delete_arrays_csv_if_exists()


def run_genetic_algorithm(config:Dict):
    """Runs the genetic algorithm to find the best weight matrix."""
    print("Running Genetic Algorithm...")
    env = WormSimulationEnv()

    GA_Class = select_ga_class(config)
    ga = GA_Class(
        population_size=config["population_size"],
        pattern=config["food_patterns"],
        total_episodes=config["total_episodes"],
        training_interval=config["training_interval"]
    )
    best_weight_matrix = ga.run(env, config["generations"])
    print("Best weight matrix found:", best_weight_matrix)


def graph_video_ngons(config):
    env = WormSimulationEnv()
    gd_video = Genetic_Dyn_Video(
        population_size=1,
        pattern=[3, 4, 5, 6, 7, 8],
        total_episodes=config["total_episodes"],
        training_interval=config["training_interval"],
    )
    gd_video.run_video_simulation(env, output_video=config.get("video_output", "food_collection_video.mp4"))

def graph_image_ngons(config):
    env = WormSimulationEnv()
    sim = Genetic_Dyn_Video(
        population_size=1,
        pattern=[3, 4, 5, 6, 7, 8],
        total_episodes=config["total_episodes"],
        training_interval=config["training_interval"],
    )
    sim.run_image_simulation()

