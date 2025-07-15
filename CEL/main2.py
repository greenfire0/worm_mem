from Worm_Env.connectome import WormConnectome
from Worm_Env.weight_dict import refresh_npz
from graphs.graph_video import Genetic_Dyn_Video
import numpy as np
from Algorithms.Pure_NOMAD import Pure_NOMAD
import ray


wc = WormConnectome(force_unit_weights=True)
print(np.unique(wc.exc.data))   # [1.]
print(np.unique(wc.inh.data))   # [1.]   (returned as −1 via wc[...] API)
print(np.unique(wc.gap.data))   # [1.]

if True:
    ray.init(
                ignore_reinit_error=True,
                object_store_memory=12 * 1024 * 1024 * 1024,
                num_cpus=16,
            )
    pn = Pure_NOMAD()
    pn.run()


gv = Genetic_Dyn_Video();gv.run_video_simulation()        # defaults: patterns 0 & 1