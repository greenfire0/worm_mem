from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import refresh_npz
from graphs.graph_video import Genetic_Dyn_Video
import numpy as np
from Algorithms.Pure_NOMAD import Pure_NOMAD
import ray




if False:
    ray.init(
                ignore_reinit_error=True,
                object_store_memory=12 * 1024 * 1024 * 1024,
                num_cpus=16,
            )
    pn = Pure_NOMAD()
    pn.run()


gv = Genetic_Dyn_Video();gv.run_video_simulation()        # defaults: patterns 0 & 1