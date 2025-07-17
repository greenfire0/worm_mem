import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Dict
from matplotlib import cm, colors as mcolors

__all__ = ["ConnectomeViewer", "ConnectomeVideoRecorder"]  # recorder unchanged


# ──────────────────────────────────────────────────────────
#  1.  Interactive network viewer (revised)
# ──────────────────────────────────────────────────────────
class ConnectomeViewer:
    """
    Visualise activity in a WormConnectome (old or enhanced version).

    • Works with *binary*, *heat*, or *energy* colour modes.
    • Automatically picks the correct state column (t0 vs. curcol).
    • Uses per‑neuron thresholds if `wc._thr_map` exists; otherwise
      falls back to a global scalar.
    """

    def __init__(self,
                 wc,
                 layout: str = "kamada_kawai",
                 threshold: Optional[float] = None,
                 max_edges: int = 6000,
                 node_size: int = 40,
                 color_mode: str = "binary",
                 colormap: str = "plasma",
                 vmax: Optional[float] = None,
                 inact_color: str = "#d3d3d3",
                 act_color: str = "#ff5555"):

        if color_mode not in ("binary", "heat", "energy"):
            raise ValueError("color_mode must be 'binary', 'heat', or 'energy'")

        self.wc         = wc
        self.node_size  = node_size
        self.color_mode = color_mode
        self.inact_col  = inact_color
        self.act_col    = act_color

        # ----- threshold map ---------------------------------------
        if threshold is not None:
            # user‑supplied scalar overrides everything
            self.thr_map = np.full(wc.N, float(threshold))
        elif hasattr(wc, "_thr_map"):
            # new enriched model: per‑neuron array
            self.thr_map = wc._thr_map.astype(float)
        elif hasattr(wc, "threshold"):
            # legacy global scalar
            self.thr_map = np.full(wc.N, float(wc.threshold))
        else:
            self.thr_map = np.full(wc.N, 30.0)  # sensible default

        # single scalar for colour normalisation
        thr_scalar = float(np.median(self.thr_map))

        # ----- colour‑map helpers ----------------------------------
        if color_mode == "heat":
            vmax = (2.0 * thr_scalar) if vmax is None else float(vmax)
            self.norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            self.cmap = cm.get_cmap(colormap)
        elif color_mode == "energy":
            vmax = thr_scalar if vmax is None else float(vmax)
            self.norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
            self.cmap = cm.get_cmap("Blues_r")

        # ----- which column holds \"current\" membrane potentials? --
        self._state_col_attr = "t0" if hasattr(wc, "t0") else "curcol"

        # ----- build lightweight NetworkX graph --------------------
        G = nx.DiGraph()
        G.add_nodes_from(wc.names)

        idx = np.argsort(np.abs(wc._edge_w))[::-1][:max_edges]
        for (layer, i, j), w in zip(np.asarray(wc._edge_ptr)[idx],
                                    wc._edge_w[idx]):
            G.add_edge(wc.names[i], wc.names[j], weight=w)
        self.G = G

        # ----- layout ----------------------------------------------
        self.pos = (self._compute_group_layout()          if layout == "groups"         else
                    self._compute_kamada_group_layout()  if layout == "kamada_groups"  else
                    self._compute_standard_layout(layout))

        # ----- initial draw ----------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        init_color = (inact_color if color_mode == "binary"
                      else self.cmap(self.norm(0.0)))
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
        col = getattr(self.wc, self._state_col_attr)
        pot = self.wc.post[:, col]

        if self.color_mode == "binary":
            active = np.abs(pot) > self.thr_map
            colors = np.where(active, self.act_col, self.inact_col)

        elif self.color_mode == "heat":
            mag    = np.abs(pot)
            colors = self.cmap(self.norm(mag))

        else:  # energy
            mag    = np.abs(pot)
            blues  = self.cmap(self.norm(mag))
            active = mag > self.thr_map
            colors = np.where(active[:, None],
                              mcolors.to_rgba(self.act_col), blues)

        self.node_coll.set_color(colors)
        self.fig.canvas.draw_idle()

    update = step  # alias for backward compatibility

    # ─────────────────────────────────────────────────────────
    #  Layout helpers (unchanged from your original version)
    # ─────────────────────────────────────────────────────────
    def _compute_standard_layout(self, name: str):
        if name == "spring":   return nx.spring_layout(self.G, seed=1)
        if name == "shell":    return nx.shell_layout(self.G)
        if name == "circular": return nx.circular_layout(self.G)
        if name == "kamada_kawai":
            H = self.G.copy()
            for _, _, d in H.edges(data=True):
                d["weight"] = abs(d["weight"])
            return nx.kamada_kawai_layout(H, weight="weight")
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
            d["weight"] = abs(d["weight"])
        pos = nx.kamada_kawai_layout(H, weight="weight")

        g_touch  = [self.wc.names[i] for i in self.wc.touch_idx]
        g_food   = [self.wc.names[i] for i in self.wc.food_idx]
        g_muscle = [self.wc.names[i] for i in np.where(self.wc.muscle_mask)[0]]
        g_other  = [n for n in self.wc.names if n not in g_touch + g_food + g_muscle]

        groups  = [g_touch, g_food, g_other, g_muscle]
        offsets = [-gap, 0.0, gap, 2 * gap]
        for g, dx in zip(groups, offsets):
            for n in g:
                pos[n][0] += dx
        return pos
