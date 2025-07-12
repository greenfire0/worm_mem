# genetic_dyn_video.py
# ──────────────────────────────────────────────────────────────────────
#  High-level visualiser
#  - Runs the *last* genome saved to Pure_nomad.csv under one or more
#    cue patterns (`patterns` argument).
#  - Captures every frame via the WormSimulationEnv.render() method.
#  - Builds a side-by-side montage for all patterns, exports MP4, and
#    removes the temporary PNGs.
# ──────────────────────────────────────────────────────────────────────
import os
import cv2
import numpy as np
from tqdm import tqdm
from moviepy import ImageSequenceClip

from Worm_Env.connectome  import WormConnectome
from Worm_Env.celegan_env import WormSimulationEnv
from util.write_read_txt  import read_arrays_from_csv_pandas


# ──────────────────────────────────────────────────────────────────────
# helper: grab RGB frame **after** env.render()
# ──────────────────────────────────────────────────────────────────────
def get_frame_from_env(env: WormSimulationEnv) -> np.ndarray:
    """
    Renders the env and returns a BGR image (H×W×3 uint8) suitable for
    OpenCV / MoviePy. Works on back-ends where only `tostring_argb`
    is available (e.g. FigureCanvasAgg).
    """
    env.render()                                   # draw current state
    canvas = env.fig.canvas
    w, h   = canvas.get_width_height()

    # -- ARGB → RGB -----------------------------------------------
    argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    argb = argb.reshape((h, w, 4))
    rgb  = argb[:, :, 1:]                          # drop alpha
    # --------------------------------------------------------------

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)    # OpenCV wants BGR


# ──────────────────────────────────────────────────────────────────────
#  Main class
# ──────────────────────────────────────────────────────────────────────
class Genetic_Dyn_Video:
    """
    Visualise one or more *trained* worms.

    Parameters
    ----------
    patterns : list[int]
        Each element is passed to `env.reset(prob)` so you can show the
        1-pulse task (prob=0), the 2-pulse task (prob=1), or any custom
        cue schedule you encoded in training.
    total_episodes : int
        How many episodes to run for each pattern.
    training_interval : int
        Number of timesteps per episode (must match env.episode_len).
    tmp_img_folder : str
        Temporary directory to store raw PNG frames.
    """

    def __init__(
        self,
        patterns           = (0, 1),   # default: both tasks
        total_episodes     = 1,
        training_interval  = 100,
        tmp_img_folder     = "tmp_img"
    ):
        self.patterns          = list(patterns)
        self.total_episodes    = total_episodes
        self.training_interval = training_interval
        self.tmp_img_folder    = tmp_img_folder
        os.makedirs(self.tmp_img_folder, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    #  video run
    # ──────────────────────────────────────────────────────────────
    def run_video_simulation(self,
                             output_video: str = "simulation.mp4",
                             fps: int = 25) -> None:

        # 1 ─ load *last* genome from Pure_nomad.csv
        rows = read_arrays_from_csv_pandas("Pure_nomad.csv")
        if not rows:
            raise FileNotFoundError("Pure_nomad.csv is empty.")
        genome = np.asarray(rows[-1], dtype=float)         # last row
        worm   = WormConnectome(init_weights=genome)

        # 2 ─ create an env for each pattern
        envs = []
        for pat in self.patterns:
            env = WormSimulationEnv(graphing=True)
            env.reset(pat)
            envs.append(env)

        # montage grid (≤ 3 columns to stay compact)
        n_env = len(envs)
        n_col = 2 if n_env == 2 else min(3, n_env)
        n_row = int(np.ceil(n_env / n_col))

        # 3 ─ simulation loop
        frame_idx = 0
        for a in [0]:
            env.reset(a)
            for _ in range(episodes):  # total_episodes
                observation = env._get_observations()
                for s in range(interval):  # training_interval
                    movement = candidate.move(observation[0], observation[4])
                    observation = env.step(movement)
                sum_rewards+=env.reward

                # pad to full grid size with blank images
                h, w, _ = frames[0].shape
                blank   = np.zeros_like(frames[0])
                while len(frames) < n_row * n_col:
                    frames.append(blank)

                # build montage rows then concat vertically
                rows_img = [
                    cv2.hconcat(frames[r*n_col:(r+1)*n_col])
                    for r in range(n_row)
                ]
                montage = cv2.vconcat(rows_img)

                cv2.imwrite(
                    os.path.join(self.tmp_img_folder,
                                 f"frame_{frame_idx:05d}.png"),
                    montage
                )
                frame_idx += 1

        # 4 ─ compile → MP4
        pngs = sorted(p for p in os.listdir(self.tmp_img_folder)
                      if p.endswith(".png"))
        clip = ImageSequenceClip(
            [os.path.join(self.tmp_img_folder, p) for p in pngs],
            fps=fps
        )
        clip.write_videofile(output_video, codec="libx264", audio=False)

        # 5 ─ clean up
        for p in pngs:
            os.remove(os.path.join(self.tmp_img_folder, p))
        print(f"✔ Saved {output_video} — {len(pngs)} frames, folder cleaned.")


# -----------------------------------------------------------------------------
# quick CLI usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    gv = Genetic_Dyn_Video()             # defaults: patterns (0,1)
    gv.run_video_simulation()
