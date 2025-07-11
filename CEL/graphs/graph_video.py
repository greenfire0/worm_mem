# genetic_dyn_video.py
# ---------------------------------------------------------------------
#  High-level visualiser: run trained genomes on two food conditions,
#  capture every frame, stitch them into one MP4, then clean up PNGs.
#  --------------------------------------------------------------------
import os, math, cv2, numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from moviepy import ImageSequenceClip
import ray                                    # only for your env/worker init
from Worm_Env.connectome     import WormConnectome
from Worm_Env.celegan_env    import WormSimulationEnv
from util.write_read_txt     import read_arrays_from_csv_pandas


# ──────────────────────────────────────────────────────────────────────
# helper: convert the Matplotlib canvas of an env → ndarray (BGR)
# ──────────────────────────────────────────────────────────────────────
def get_frame_from_env(sim_env):
    sim_env.ax.clear()
    worm = sim_env.worm

    if worm.sees_food:
        color = "c"
    else:
        color = "ro"
    sim_env.ax.plot(worm.position[0], worm.position[1], color)
    sim_env.ax.plot([worm.position[0],
                     worm.position[0] + 100 * np.cos(worm.facing_dir)],
                    [worm.position[1],
                     worm.position[1] + 100 * np.sin(worm.facing_dir)], 'b-')

    
    sim_env.ax.set_xlim(0, sim_env.dimx)
    sim_env.ax.set_ylim(0, sim_env.dimy)
    sim_env.fig.canvas.draw()

    w, h = sim_env.fig.canvas.get_width_height()
    buf  = np.frombuffer(sim_env.fig.canvas.tostring_argb(), dtype=np.uint8)
    buf  = buf.reshape((h, w, 4))
    rgb  = buf[:, :, 1:]                 # drop alpha
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────────────────
#  Main class
# ──────────────────────────────────────────────────────────────────────
class Genetic_Dyn_Video:
    """
    • Loads the **last genome** from array0.csv and array1.csv (default).
    • Runs both worms side-by-side.
    • Saves simulation.mp4 and removes every frame PNG.
    """

    def __init__(self,
                 patterns          = [0],          # prob list
                 total_episodes    = 1,
                 training_interval = 100,
                 tmp_img_folder    = "tmp_img"):

        self.patterns           = list(patterns)
        self.total_episodes     = total_episodes
        self.training_interval  = training_interval
        self.tmp_img_folder     = tmp_img_folder
        os.makedirs(self.tmp_img_folder, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    #  video run
    # ──────────────────────────────────────────────────────────────
    def run_video_simulation(self,
                             output_video="simulation.mp4",
                             fps=25):

        # 1 ─ load genomes + envs
        candidates, envs = [], []
        for pat in self.patterns:
            rows   = read_arrays_from_csv_pandas(f"Pure_nomad.csv")
            if not rows:
                raise FileNotFoundError(f"array{pat}.csv has no genome rows.")
            genome = np.asarray(rows[-1], dtype=float)
            candidates.append(WormConnectome(init_weights=genome))

            env:WormSimulationEnv = WormSimulationEnv(graphing=True)
            env.reset(pat)
            envs.append(env)

        steps       = self.total_episodes * self.training_interval
        done        = [False]*len(envs)
        last_frames = [None ]*len(envs)

        # calculate montage grid (rows×cols)
        n_env = len(envs)
        n_col = 2 if n_env == 2 else min(3, n_env)
        n_row = int(np.ceil(n_env / n_col))
        rewards = 0
        # 2 ─ simulation loop
        for step in tqdm(range(steps), desc="Simulating"):
            frames = []
            for i, (env, wc) in enumerate(zip(envs, candidates)):
                if not done[i]:
                    obs   = env._get_observations()
                    move  = wc.move(obs[0], obs[4])
                    rewards+=  env.step(move)[1]

                frames.append(last_frames[i] if done[i]
                              else get_frame_from_env(env))
            # pad with blanks
            h, w, _ = frames[0].shape
            blank   = np.zeros_like(frames[0])
            while len(frames) < n_row*n_col:
                frames.append(blank)

            # build montage
            rows = [cv2.hconcat(frames[r*n_col:(r+1)*n_col]) for r in range(n_row)]
            montage = cv2.vconcat(rows)

            cv2.imwrite(os.path.join(self.tmp_img_folder,
                                     f"frame_{step:05d}.png"),
                        montage)
        print(rewards)

        # 3 ─ compile → MP4
        pngs  = sorted(p for p in os.listdir(self.tmp_img_folder)
                       if p.endswith(".png"))
        clip  = ImageSequenceClip([os.path.join(self.tmp_img_folder, p) for p in pngs],
                                  fps=fps)
        clip.write_videofile(output_video, codec="libx264", audio=False)

        # 4 ─ clean up PNGs
        for p in pngs:
            os.remove(os.path.join(self.tmp_img_folder, p))
        print(f"✔  Saved {output_video}  –  {len(pngs)} frames, folder cleaned.")


# -----------------------------------------------------------------------------
# quick CLI usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    gv = Genetic_Dyn_Video()        # defaults: patterns 0 & 1
    gv.run_video_simulation()
