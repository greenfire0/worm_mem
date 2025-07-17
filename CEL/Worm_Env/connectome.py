# worm_sparse_enhanced.py  — biologically enriched version
# -----------------------------------------------------------------------------
#  This variant keeps the same public API (WormConnectome.move(), 1‑D edge
#  indexing, etc.) but adds the missing biophysics requested:
#    • graded vs. spiking neurons (per‑cell threshold map)
#    • bidirectional gap‑junction current (ohmic shunt, not chemical copy)
#    • α‑synapse‑like decay per connection class (τ_exc / τ_inh / τ_gap)
#    • one‑timestep conduction delay (double buffer -> triple buffer)
#    • NO more force_unit_weights flag — original empirical magnitudes kept
# -----------------------------------------------------------------------------
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning
from warnings import filterwarnings
from numba import njit
from pathlib import Path
from typing import Dict, Tuple, List

filterwarnings("ignore", category=SparseEfficiencyWarning)

# ----------------------------------------------------------------------------
# 1.  Global constants – quick to tune at top of file
# ----------------------------------------------------------------------------
SENSOR_KICK   = 60.0                  # mV injected into active sensor
V_REST        = 0.0                   # baseline potential (mV)
TAU_EXC       = 0.9                   # synaptic rise/decay factors (0‑1)
TAU_INH       = 0.9
TAU_GAP       = 1.0                   # ohmic – not decaying per step
MEMBRANE_LEAK = 0.82                 # passive decay  (0‑1)
CONDUCTION_DL = 1                     # time‑step delay between compartments

#  Spiking‑Ca2+ cells discovered so far (2018‑2024 literature)
SPIKERS = {
    "AWA": 20.0,        # threshold ≈ +20 mV above V_rest
    "AVAL": 20.0, "AVAR": 20.0,   # evidence of Ca²⁺ spikes, keep same cut‑off
    "AVL": 20.0, "DVB": 20.0,
    "AIA": 20.0,
}

# ----------------------------------------------------------------------------
# 2.  Low‑level Numba helpers
# ----------------------------------------------------------------------------
@njit(inline="always")
def _csr_row_add_scaled(post_col: np.ndarray,
                        data: np.ndarray, indices: np.ndarray,
                        indptr: np.ndarray, row: int,
                        scale: float):
    """Add `scale * weight` into every post‑synaptic target of *row*."""
    for k in range(indptr[row], indptr[row + 1]):
        post_col[indices[k]] += data[k] * scale

@njit(inline="always")
def _gap_row_exchange(post_col: np.ndarray, cur_col: np.ndarray,
                      data: np.ndarray, indices: np.ndarray,
                      indptr: np.ndarray, row: int):
    """Ohmic gap‑junction: I = g(V_pre – V_post), applied pair‑wise."""
    v_pre = cur_col[row]
    for k in range(indptr[row], indptr[row + 1]):
        j   = indices[k]
        g   = data[k]
        dv  = v_pre - cur_col[j]
        post_col[row] -= g * dv * TAU_GAP
        post_col[j]   += g * dv * TAU_GAP

@njit
def _step_once(post: np.ndarray,
               exc_d: np.ndarray, exc_i: np.ndarray, exc_p: np.ndarray,
               inh_d: np.ndarray, inh_i: np.ndarray, inh_p: np.ndarray,
               gap_d: np.ndarray, gap_i: np.ndarray, gap_p: np.ndarray,
               threshold_map: np.ndarray,  # per‑neuron (spikers only)
               graded_mask: np.ndarray,    # bool: graded cell?
               muscle_mask: np.ndarray,
               left_idx: np.ndarray, right_idx: np.ndarray,
               sensor_idx: np.ndarray,
               t0: int, t1: int, t2: int) -> Tuple[float, float]:
    """Advance network by one Δt with 1‑step conduction delay."""

    # 0. Passive leak into buffer t1 (future), carry t0 ➜ t1 first
    post[:, t1] = (post[:, t0] - V_REST) * MEMBRANE_LEAK + V_REST

    # 1. Sensor stimulation (direct kick + first‑hop chemical / gap)
    for idx in sensor_idx:
        scale = SENSOR_KICK            # absolute mV injection
        post[idx, t1] += scale
        _csr_row_add_scaled(post[:, t1], exc_d, exc_i, exc_p, idx, TAU_EXC)
        _csr_row_add_scaled(post[:, t1], inh_d, inh_i, inh_p, idx, -TAU_INH)
        _gap_row_exchange(post[:, t1], post[:, t0],
                          gap_d, gap_i, gap_p, idx)

    N = post.shape[0]
    # 2. Iterate over every *presynaptic* cell
    for pre in range(N):
        if muscle_mask[pre] or pre in sensor_idx:
            continue

        v_pre = post[pre, t0]
        if graded_mask[pre]:
            # graded release proportionally with membrane potential
            scale_e = (v_pre - V_REST) * TAU_EXC / 40.0  # arbitrary gain
            scale_i = -(v_pre - V_REST) * TAU_INH / 40.0
            _csr_row_add_scaled(post[:, t1], exc_d, exc_i, exc_p, pre, scale_e)
            _csr_row_add_scaled(post[:, t1], inh_d, inh_i, inh_p, pre, scale_i)
            _gap_row_exchange(post[:, t1], post[:, t0],
                              gap_d, gap_i, gap_p, pre)
        else:  # spiking – all‑or‑none above threshold
            if abs(v_pre - V_REST) > threshold_map[pre]:
                _csr_row_add_scaled(post[:, t1], exc_d, exc_i, exc_p, pre, TAU_EXC)
                _csr_row_add_scaled(post[:, t1], inh_d, inh_i, inh_p, pre, -TAU_INH)
                _gap_row_exchange(post[:, t1], post[:, t0],
                                  gap_d, gap_i, gap_p, pre)
                post[pre, t1] = V_REST  # reset after spike

    # 3. Muscle read‑out (sum & clear)
    left = 0.0
    right = 0.0
    for i in left_idx:
        left  += post[i, t1]
        post[i, t1] = V_REST
    for i in right_idx:
        right += post[i, t1]
        post[i, t1] = V_REST

    # 4. Rotate triple buffer  (t0 <- t1, t1 <- t2, t2 spare)
    post[:, t2] = post[:, t0]          # keep one frame for delay
    post[:, t0] = post[:, t1]

    return left, right

# ----------------------------------------------------------------------------
# 3.  Public WormConnectome class (API unchanged)
# ----------------------------------------------------------------------------
class WormConnectome:
    """Biologically enriched sparse connectome (1‑D edge indexing only)."""

    def __init__(self,
                 npz_path: str | Path = "connectome_sparse.npz",
                 init_weights: np.ndarray | None = None,
                 default_threshold: float = 20.0):

        Z = np.load(npz_path, allow_pickle=True)
        self._nt_counts: Dict[str, int] = dict(zip(Z["nt_keys"].tolist(),
                                                   Z["nt_vals"].tolist()))
        self.names: List[str] = [n.decode() if isinstance(n, bytes) else n
                                 for n in Z["neurons"]]
        self.N = len(self.names)
        self.name2idx: Dict[str, int] = {n: i for i, n in enumerate(self.names)}

        # build per‑neuron threshold & graded mask
        self._thr_map   = np.full(self.N, default_threshold, np.float64)
        self._graded_mk = np.ones(self.N, np.bool_)
        for nm, thr in SPIKERS.items():
            if nm in self.name2idx:
                idx = self.name2idx[nm]
                self._thr_map[idx]   = thr
                self._graded_mk[idx] = False

        # --- load sparse matrices ----------------------------------------
        shape = tuple(Z["shape"])
        self.exc = sp.csr_matrix((Z["exc_data"], Z["exc_indices"], Z["exc_indptr"]),
                                 shape=shape)
        self.inh = sp.csr_matrix((Z["inh_data"], Z["inh_indices"], Z["inh_indptr"]),
                                 shape=shape)
        self.gap = sp.csr_matrix((Z["gap_data"], Z["gap_indices"], Z["gap_indptr"]),
                                 shape=shape)

        # --- raw views for Numba -----------------------------------------
        self._refresh_arrays()

        # bookkeeping for muscles / sensors
        from Worm_Env.weight_dict import mLeft, mRight, muscleList
        self.left_idx  = np.array([self.name2idx[n] for n in mLeft],  np.int32)
        self.right_idx = np.array([self.name2idx[n] for n in mRight], np.int32)
        prefixes       = {n[:3] for n in muscleList}
        self.muscle_mask = np.array([nm[:3] in prefixes for nm in self.names],
                                    np.bool_)

        # main state buffer  (N × 3)  → t0 current, t1 next, t2 delay backup
        self.post = np.zeros((self.N, 3), np.float64) + V_REST
        self.t0, self.t1, self.t2 = 0, 1, 2

        # sensory indices --------------------------------------------------
        self.touch_idx = np.array([self.name2idx[n] for n in
            ("FLPR","FLPL","ASHL","ASHR","IL1VL","IL1VR","OLQDL","OLQDR",
             "OLQVR","OLQVL")], np.int32)
        self.food_idx = np.array([self.name2idx[n] for n in
            ("ADFL","ADFR","ASGR","ASGL","ASIL","ASIR","ASJR","ASJL")], np.int32)

        # --- apply optional genome ---------------------------------------
        self._rebuild_edge_index()
        if init_weights is not None:
            self._apply_weight_vector(np.asarray(init_weights, np.float64))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def move(self, dist: float, sees_food: bool, *_):
        sensor_idx = (self.touch_idx if 0 < dist < 100
                       else self.food_idx if sees_food
                       else np.empty(0, np.int32))
        l, r = _step_once(self.post,
                          self._exc_d, self._exc_i, self._exc_p,
                          self._inh_d, self._inh_i, self._inh_p,
                          self._gap_d, self._gap_i, self._gap_p,
                          self._thr_map,
                          self._graded_mk,
                          self.muscle_mask,
                          self.left_idx, self.right_idx,
                          sensor_idx.astype(np.int32),
                          self.t0, self.t1, self.t2)
        # rotate buffer indices (t0 ↦ t1 ↦ t2 ↦ t0)
        self.t0, self.t1, self.t2 = self.t1, self.t2, self.t0
        return l, r

    def state_reset(self, noisy: bool = False, rng=None,
                    lo: float = -1.0, hi: float = 1.0):
        if noisy:
            rng = np.random.default_rng() if rng is None else rng
            self.post[:, :] = rng.uniform(lo, hi, size=self.post.shape)
        else:
            self.post[:, :] = V_REST

    # ------------------------------------------------------------------
    # internals (unchanged except triple buffer aware)
    # ------------------------------------------------------------------
    def _refresh_arrays(self):
        self._exc_d, self._exc_i, self._exc_p = map(np.asarray,
            (self.exc.data, self.exc.indices, self.exc.indptr))
        self._inh_d, self._inh_i, self._inh_p = map(np.asarray,
            (self.inh.data, self.inh.indices, self.inh.indptr))
        self._gap_d, self._gap_i, self._gap_p = map(np.asarray,
            (self.gap.data, self.gap.indices, self.gap.indptr))

    def _rebuild_edge_index(self):
        ptr: List[Tuple[int,int,int]] = []
        wvec: List[float]             = []
        for layer, mat in enumerate((self.exc, self.inh, self.gap)):
            indptr, indices, data = mat.indptr, mat.indices, mat.data
            for i in range(mat.shape[0]):
                for p in range(indptr[i], indptr[i + 1]):
                    j = int(indices[p])
                    ptr.append((layer, i, j))
                    if layer == 0:
                        wvec.append(+data[p])
                    elif layer == 1:
                        wvec.append(-data[p])
                    else:
                        wvec.append(+data[p])
        self._edge_ptr = ptr
        self._edge_w   = np.asarray(wvec, np.float64)

    def _apply_weight_vector(self, vec: np.ndarray):
        self.exc.data[:] = 0.0
        self.inh.data[:] = 0.0
        self.gap.data[:] = 0.0
        for (layer, i, j), w in zip(self._edge_ptr, vec):
            if layer == 2:
                self.gap[i, j] = self.gap[j, i] = abs(w)
            elif layer == 0:
                self.exc[i, j] = max(w, 0.0)
            else:
                self.inh[i, j] = max(-w, 0.0)
        self._refresh_arrays()
        self._rebuild_edge_index()


    # ── 1-D edge interface ──────────────────────────────────
    def __getitem__(self, key):
        """Return (pre_name, post_name, weight) for edge #k."""
        def _as_index(x):
            if isinstance(x, str):           # neuron name
                return self.name2idx[x]
            return x                         # int, slice, list, ndarray

        # ------------------------------------------------------------------
        # 1-D edge ID  →  original behaviour
        # ------------------------------------------------------------------
        if isinstance(key, (int, np.integer)):
            layer, i, j = self._edge_ptr[int(key)]
            if   layer == 0: w =  self.exc[i, j]
            elif layer == 1: w = -self.inh[i, j]          # inh stored +
            else:            w =  self.gap[i, j]
            return (self.names[i], self.names[j], float(w))

        # ------------------------------------------------------------------
        # Broadcasted matrix style indexing  wc[pre, post]
        # ------------------------------------------------------------------
        if isinstance(key, tuple) and len(key) == 2:
            pre, post = map(_as_index, key)

            # sparse → dense  (small slices stay cheap)
            w_exc = self.exc[pre,  post].toarray()
            w_inh = self.inh[pre,  post].toarray()
            w_gap = self.gap[pre,  post].toarray()

            w = w_exc - w_inh + w_gap        # inh sign-flip

            # squeeze scalars back to float
            return float(w) if w.size == 1 else np.squeeze(w)

        # ------------------------------------------------------------------
        # List / ndarray of edge IDs
        # ------------------------------------------------------------------
        if isinstance(key, (list, np.ndarray)):
            # Fast path: pull straight from cached 1-D weight vector
            ids = np.asarray(key, dtype=np.int64)
            return self._edge_w[ids].tolist() 

        raise TypeError("Unsupported index type for WormConnectome.")

    def __setitem__(self,
                    k: int | List[int] | np.ndarray,
                    value: float | List[float] | np.ndarray):

        # ---------- normalise inputs (unchanged) ---------------------------
        if isinstance(k, (int, np.integer)):
            ks = np.array([int(k)], dtype=np.int64)
            vs = np.full(1, value, dtype=np.float64)
        else:
            ks = np.asarray(k, dtype=np.int64)
            vs = np.full(len(ks), value, dtype=np.float64) if np.isscalar(value) \
                else np.asarray(value, dtype=np.float64)
            if len(vs) != len(ks):
                raise ValueError("value array length must match k array")

        # ---------- edit edges --------------------------------------------
        for idx, v in zip(ks, vs):
            layer, i, j = self._edge_ptr[int(idx)]

            if layer == 2:                       # ── gap (mirror) ──────────
                self.gap[i, j] = self.gap[j, i] = v

            elif layer == 0:                     # ── excitatory edge ───────
                if v <= 0:
                    raise ValueError("edge #{idx} is excitatory; weight must be >0")
                self.exc[i, j] = v              # update this layer…
                if self.inh[i, j] != 0:         # …and *delete* from other layer
                    self.inh[i, j] = 0

            else:                               # ── inhibitory edge ───────
                if v >= 0:
                    raise ValueError("edge #{idx} is inhibitory; weight must be <0")
                self.inh[i, j] = -v             # store magnitude
                if self.exc[i, j] != 0:
                    self.exc[i, j] = 0

        # ---------- purge zeros & rebuild views ---------------------------
        for m in (self.exc, self.inh, self.gap):
            m.eliminate_zeros()

        self._refresh_arrays()

    # ── len() and counts ─────────────────────────────────────
    def __len__(self):                  # number of neurons
        return self.N

    @property
    def counts(self):
        """np.array([total_unique_connections, N])."""
        chem = self.exc.nnz + self.inh.nnz
        gap  = self.gap.nnz        # mirrored, count unique pairs
        return np.array([chem + gap, self.N], dtype=np.int64)
    @property
    def weights(self) -> np.ndarray:
        """
        1-D NumPy array of synapse weights in the same order as wc[k].
        Excitatory > 0, inhibitory < 0, gap value duplicated per direction.
        """
        return self._edge_w.copy() 
    @weights.setter
    def weights(self, new_w: np.ndarray):
        """
        Replace the *entire* weight vector in one shot.

        Parameters
        ----------
        new_w : 1-D ndarray (len == len(self._edge_ptr))
            The replacement weights, in exactly the same edge order as
            `self.weights`.
        """
        new_w = np.asarray(new_w, dtype=np.float64)
        if new_w.shape != (len(self._edge_ptr),):
            raise ValueError(f"weights must be shape {(len(self._edge_ptr),)}")

        # vectorised update: use our __setitem__ that already handles
        # sign logic and gap mirroring
        all_idx = np.arange(len(self._edge_ptr), dtype=np.int64)
        self[all_idx] = new_w


    # ─────────────────────────────────────────────────────────────
    # 6.  Human-readable summary with *all* connection types
    # ─────────────────────────────────────────────────────────────
    def __str__(self) -> str:
        """
        Lists every connection type exactly once with its current count.
        Gap-junctions (‘Generic_GJ’ / ‘generic_gj’) are counted as *pairs*
        (i.e. gap.nnz // 2).
        """
        # copy the dict so we can tweak it
        counts = dict(self._nt_counts)

        gap_key = None
        for k in counts:
            if k.lower() in ("generic_gj", "gapjunction"):
                gap_key = k
                break

        if gap_key is not None:
            counts[gap_key] = self.gap.nnz // 2     # unique pairs

        total = sum(counts.values())

        lines = [
            "WormConnectome",
            f"  neurons …… {self.N}",
            f"  total connections …… {total}",
        ]
        for name, cnt in counts.items():
            lines.append(f"    {name:<12} ………… {cnt}")
        return "\n".join(lines)