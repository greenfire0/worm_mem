import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Dict, Tuple
from matplotlib import cm, colors as mcolors

__all__ = ["ConnectomeViewer", "ConnectomeVideoRecorder"]

# ──────────────────────────────────────────────────────────
#  1.  Interactive network viewer
# ──────────────────────────────────────────────────────────

class ConnectomeViewer:
    """Live visualiser for a *WormConnectome*.

    A coloured node map is updated every simulation step. Two colour modes
    are available:

    * **binary**  – classic grey/red where *red* means
      ``|V| > threshold``.
    * **heat**    – continuous colour‑map (e.g. *plasma*) whose intensity
      scales with ``|V|``. Negative voltages are folded onto the positive
      axis (absolute value) so inhibition shows up just as strongly.

    Parameters
    ----------
    wc : WormConnectome
        The **same** connectome object your simulation mutates.
    layout : {'kamada_kawai', 'spring', 'shell', 'circular',
              'groups', 'kamada_groups'}, default 'kamada_kawai'
        Graph arrangement algorithm.  ``groups`` and ``kamada_groups``
        push *touch → food → other → muscle* left‑to‑right.
    threshold : float, optional
        Binary‑mode cut‑off. Default: ``wc.threshold``.
    max_edges : int, default 6000
        Only the heaviest *max_edges* connections are drawn.
    node_size : int, default 40
    color_mode : {'binary', 'heat', 'energy'}, default 'binary'
    colormap : str, default 'plasma'  (ignored for 'energy'; uses 'Blues')
        Matplotlib c‑map name used when ``color_mode='heat'``.
    vmax : float, optional
        Upper bound for colour normalisation (heat mode).  Defaults to
        ``2*threshold``.
    inact_color : str, default '#d3d3d3'
        Node colour when inactive (binary mode only).
    act_color : str, default '#ff5555'
        Active colour when ``color_mode='binary'``.
    """

    def __init__(self,
                 wc,
                 layout: str = 'kamada_kawai',
                 threshold: Optional[float] = None,
                 max_edges: int = 6000,
                 node_size: int = 40,
                 color_mode: str = 'binary',
                 colormap: str = 'plasma',
                 vmax: Optional[float] = None,
                 inact_color: str = '#d3d3d3',
                 act_color: str = '#ff5555'):

        if color_mode not in ('binary', 'heat', 'energy'):
            raise ValueError("color_mode must be 'binary' or 'heat'")

        self.wc         = wc
        self.thr        = wc.threshold if threshold is None else float(threshold)
        self.node_size  = node_size
        self.color_mode = color_mode
        self.inact_col  = inact_color
        self.act_col    = act_color

        # Heat‑map helpers -------------------------------------------
        if color_mode == 'heat':
            vmax = 2.0 * self.thr if vmax is None else float(vmax)
            self.norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            self.cmap = cm.get_cmap(colormap)
        elif color_mode == 'energy':
            vmax = 1.0 * self.thr if vmax is None else float(vmax)
            self.norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            self.cmap = cm.get_cmap('Blues_r')

        # ── build NetworkX graph ───────────────────────────────────
        G = nx.DiGraph()
        G.add_nodes_from(wc.names)

        idx = np.argsort(np.abs(wc._edge_w))[::-1][:max_edges]
        for (layer, i, j), w in zip(np.asarray(wc._edge_ptr)[idx], wc._edge_w[idx]):
            G.add_edge(wc.names[i], wc.names[j], weight=w)
        self.G = G

        # ── layout --------------------------------------------------
        self.pos = (self._compute_group_layout()          if layout == 'groups'         else
                     self._compute_kamada_group_layout() if layout == 'kamada_groups' else
                     self._compute_standard_layout(layout))

        # ── static drawing -----------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        if color_mode == 'binary':
            init_color = self.inact_col
        elif color_mode == 'heat':
            init_color = self.cmap(self.norm(0.0))
        else:  # energy
            init_color = self.cmap(self.norm(0.0))
        self.node_coll = nx.draw_networkx_nodes(
            G, self.pos, node_size=node_size,
            node_color=init_color, ax=self.ax
        )
        nx.draw_networkx_edges(G, self.pos, ax=self.ax,
                               arrows=False, alpha=0.25, width=0.3)
        self.ax.set_axis_off()
        self.fig.tight_layout()
        plt.show(block=False)

    # ── public API --------------------------------------------------
    def step(self):
        """Refresh node colours – call once per simulation step."""
        pot = self.wc.post[:, self.wc.curcol]
        if self.color_mode == 'binary':
            active = np.abs(pot) > self.thr
            colors = np.where(active, self.act_col, self.inact_col)
        elif self.color_mode == 'heat':
            mag = np.abs(pot)
            colors = self.cmap(self.norm(mag))
        else:  # energy mode – blue gradient, red override
            mag = np.abs(pot)
            blues = self.cmap(self.norm(mag))
            active = mag > self.thr
            colors = np.where(active[:, None], mcolors.to_rgba(self.act_col), blues)
        self.node_coll.set_color(colors)
        self.fig.canvas.draw_idle()

    update = step  # alias

    # ─────────────────────────────────────────────────────────----
    #  Layout helpers
    # ─────────────────────────────────────────────────────────----
    def _compute_standard_layout(self, name: str):
        if name == 'spring':   return nx.spring_layout(self.G, seed=1)
        if name == 'shell':    return nx.shell_layout(self.G)
        if name == 'circular': return nx.circular_layout(self.G)
        if name == 'kamada_kawai':
            H = self.G.copy()
            for _, _, d in H.edges(data=True):
                d['weight'] = abs(d['weight'])
            return nx.kamada_kawai_layout(H, weight='weight')
        raise ValueError(f"Unknown layout '{name}'")

    def _compute_group_layout(self):
        names   = self.wc.names
        touch   = list(self.wc.touch_idx)
        food    = list(self.wc.food_idx)
        muscle  = np.where(self.wc.muscle_mask)[0].tolist()
        special = set(touch + food + muscle)
        other   = [i for i in range(self.wc.N) if i not in special]

        groups = [touch, food, other, muscle]
        x_pos  = [0, 2, 4, 6]
        pos: Dict[str, np.ndarray] = {}
        for gx, idx_list in zip(x_pos, groups):
            for rank, i in enumerate(sorted(idx_list)):
                pos[names[i]] = np.array([gx, -rank], float)
        rng = np.random.default_rng(42)
        for p in pos.values():
            p += rng.uniform(-0.05, 0.05, 2)
        return pos

    def _compute_kamada_group_layout(self, gap: float = 3.0):
        H = self.G.copy()
        for _, _, d in H.edges(data=True):
            d['weight'] = abs(d['weight'])
        pos = nx.kamada_kawai_layout(H, weight='weight')

        g_touch  = [self.wc.names[i] for i in self.wc.touch_idx]
        g_food   = [self.wc.names[i] for i in self.wc.food_idx]
        g_muscle = [self.wc.names[i] for i in np.where(self.wc.muscle_mask)[0]]
        g_other  = [n for n in self.wc.names if n not in g_touch + g_food + g_muscle]

        groups  = [g_touch, g_food, g_other, g_muscle]
        offsets = [-gap, 0.0, gap, 2*gap]
        for g, dx in zip(groups, offsets):
            for n in g:
                pos[n][0] += dx
        return pos


# ──────────────────────────────────────────────────────────
#  2.  Video recorder (unchanged)
# ──────────────────────────────────────────────────────────

import os, cv2
from tqdm import tqdm
from moviepy import ImageSequenceClip
from typing import Callable

__all__.append("ConnectomeVideoRecorder")


def _frame_from_viewer(viewer):
    canvas = viewer.fig.canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    rgb  = argb[:, :, 1:]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class ConnectomeVideoRecorder:
    def __init__(self, viewer, tmp_img_folder: str = 'tmp_img', fps: int = 25):
        self.viewer  = viewer
        self.tmp_dir = tmp_img_folder
        self.fps     = fps
        os.makedirs(self.tmp_dir, exist_ok=True)

    def record(self, simulate_step: Callable[[int], None], n_steps: int,
               output_video: str = 'connectome_activity.mp4', pause: float = 0.0):
        png = os.path.join
        frame_idx = 0
        for step in tqdm(range(n_steps), desc='Recording'):
            simulate_step(step)
            frame = _frame_from_viewer(self.viewer)
            cv2.imwrite(png(self.tmp_dir, f'frame_{frame_idx:05d}.png'), frame)
            frame_idx += 1
            if pause:
                plt.pause(pause)

        pngs = sorted(p for p in os.listdir(self.tmp_dir) if p.endswith('.png'))
        clip = ImageSequenceClip([png(self.tmp_dir, p) for p in pngs], fps=self.fps)
        clip.write_videofile(output_video, codec='libx264', audio=False)

        for p in pngs:
            os.remove(png(self.tmp_dir, p))
        print(f"✔ Saved {output_video}  ({len(pngs)} frames, {self.fps} fps)")
