from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import refresh_npz
from graphs.graph_video import Genetic_Dyn_Video
import numpy as np
from Algorithms.Pure_NOMAD import Pure_NOMAD
import ray

#gv = Genetic_Dyn_Video()        # defaults: patterns 0 & 1
#gv.run_video_simulation()



ray.init(
            ignore_reinit_error=True,
            object_store_memory=8 * 1024 * 1024 * 1024,
            num_cpus=8,
        )
pn = Pure_NOMAD()
pn.run()