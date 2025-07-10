from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import refresh_npz
import numpy as np
from Algorithms.Pure_NOMAD import Pure_NOMAD
import ray
ray.init(
            ignore_reinit_error=True,
            object_store_memory=8 * 1024 * 1024 * 1024,
            num_cpus=8,
        )
pn = Pure_NOMAD()
pn.run()