# worm_sparse.py  —  only 1-D edge indexing
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning
from warnings import filterwarnings
from numba import njit
from pathlib import Path
from typing import Dict, Tuple, List

filterwarnings("ignore", category=SparseEfficiencyWarning)   # silence CSR edit warnings

# ──────────────────────────────────────────────────────────────
# 1.  Low-level helpers
# ──────────────────────────────────────────────────────────────
@njit(inline='always')
def _csr_row_add(post_col: np.ndarray,
                 data: np.ndarray, indices: np.ndarray,
                 indptr: np.ndarray, row: int):
    for k in range(indptr[row], indptr[row + 1]):
        post_col[indices[k]] += data[k]

@njit
def _step_once(post: np.ndarray,
               exc_d: np.ndarray, exc_i: np.ndarray, exc_p: np.ndarray,
               inh_d: np.ndarray, inh_i: np.ndarray, inh_p: np.ndarray,
               gap_d: np.ndarray, gap_i: np.ndarray, gap_p: np.ndarray,
               threshold: float,
               muscle_mask: np.ndarray,
               left_idx: np.ndarray, right_idx: np.ndarray,
               sensory_idx: np.ndarray,
               cur: int, nxt: int) -> Tuple[float, float]:

    for idx in sensory_idx:                          # sensory injection
        _csr_row_add(post[:, nxt], exc_d, exc_i, exc_p, idx)
        _csr_row_add(post[:, nxt], inh_d, inh_i, inh_p, idx)
        _csr_row_add(post[:, nxt], gap_d, gap_i, gap_p, idx)

    N = post.shape[0]                                # one-hop propagation
    for pre in range(N):
        if muscle_mask[pre]:
            continue
        if abs(post[pre, cur]) > threshold:
            _csr_row_add(post[:, nxt], exc_d, exc_i, exc_p, pre)
            _csr_row_add(post[:, nxt], inh_d, inh_i, inh_p, pre)
            _csr_row_add(post[:, nxt], gap_d, gap_i, gap_p, pre)
            post[pre, nxt] = 0.0

    left = 0.0                                       # muscle read-out
    right = 0.0
    for i in left_idx:
        left  += post[i, nxt]
        post[i, nxt] = 0.0
    for i in right_idx:
        right += post[i, nxt]
        post[i, nxt] = 0.0

    post[:, cur] = post[:, nxt]                      # swap buffers
    return left, right

# ──────────────────────────────────────────────────────────────
# 2.  High-level class
# ──────────────────────────────────────────────────────────────
class WormConnectome:
    """
    Sparse connectome with **1-D edge indexing only**:
        wc[k]          -> (pre_name, post_name, weight)
        wc[k] = new_w  -> edit existing edge (gap mirrored automatically)
    """

    # ── constructor ──────────────────────────────────────────
    def __init__(self,
                 npz_path: str | Path = "connectome_sparse.npz",
                 threshold: float = 30.0,
                 init_weights: np.ndarray | None = None):

        Z                     = np.load(npz_path, allow_pickle=True)
        self._nt_counts: Dict[str, int] = dict(
            zip(Z['nt_keys'].tolist(), Z['nt_vals'].tolist())
        )
        self.names: List[str] = [n.decode() if isinstance(n, bytes) else n
                                 for n in Z['neurons']]
        self.N      = len(self.names)
        self.threshold = threshold
        self.name2idx: Dict[str, int] = {n: i for i, n in enumerate(self.names)}

        shape = tuple(Z['shape'])
        self.exc = sp.csr_matrix((Z['exc_data'], Z['exc_indices'], Z['exc_indptr']),
                                 shape=shape)
        self.inh = sp.csr_matrix((Z['inh_data'], Z['inh_indices'], Z['inh_indptr']),
                                 shape=shape)
        self.gap = sp.csr_matrix((Z['gap_data'], Z['gap_indices'], Z['gap_indptr']),
                                 shape=shape)

        # raw arrays for Numba kernels
        self._exc_d, self._exc_i, self._exc_p = map(np.asarray,
            (self.exc.data, self.exc.indices, self.exc.indptr))
        self._inh_d, self._inh_i, self._inh_p = map(np.asarray,
            (self.inh.data, self.inh.indices, self.inh.indptr))
        self._gap_d, self._gap_i, self._gap_p = map(np.asarray,
            (self.gap.data, self.gap.indices, self.gap.indptr))

        # bookkeeping for muscles / sensors (unchanged)
        from Worm_Env.weight_dict import mLeft, mRight, muscleList
        self.left_idx  = np.array([self.name2idx[n] for n in mLeft],  np.int32)
        self.right_idx = np.array([self.name2idx[n] for n in mRight], np.int32)
        prefixes       = {n[:3] for n in muscleList}
        self.muscle_mask = np.array([nm[:3] in prefixes for nm in self.names],
                                    np.bool_)

        self.post   = np.zeros((self.N, 2), np.float64)
        self.curcol = 0
        self.nextcol = 1

        self.touch_idx = np.array([self.name2idx[n] for n in
            ("FLPR","FLPL","ASHL","ASHR","IL1VL","IL1VR","OLQDL","OLQDR",
             "OLQVR","OLQVL")], np.int32)
        self.food_idx = np.array([self.name2idx[n] for n in
            ("ADFL","ADFR","ASGR","ASGL","ASIL","ASIR","ASJR","ASJL")], np.int32)

        self._rebuild_edge_index()               # build 1-D pointer list
        if init_weights is not None:
            if len(init_weights) != len(self._edge_ptr):
                raise ValueError("init_weights length mismatch: "
                                f"{len(init_weights)} vs {len(self._edge_ptr)}")
            self._apply_weight_vector(np.asarray(init_weights, np.float64))

    # ── main simulation step ─────────────────────────────────
    def move(self, dist: float, sees_food: bool, *_):
        sensory_idx = (self.touch_idx if 0 < dist < 100
                       else self.food_idx if sees_food
                       else np.empty(0, np.int32))
        left, right = _step_once(self.post,
                                 self._exc_d, self._exc_i, self._exc_p,
                                 self._inh_d, self._inh_i, self._inh_p,
                                 self._gap_d, self._gap_i, self._gap_p,
                                 self.threshold,
                                 self.muscle_mask,
                                 self.left_idx, self.right_idx,
                                 sensory_idx,
                                 self.curcol, self.nextcol)
        self.curcol, self.nextcol = self.nextcol, self.curcol
        return left, right

    # ── internal maintenance ────────────────────────────────
    def _rebuild_edge_index(self):
        """
        Build two parallel lists:
            • self._edge_ptr[k] = (layer, i, j)
            • self._edge_w[k]   = signed weight
        layer: 0 = exc , 1 = inh , 2 = gap
        """
        ptr: List[Tuple[int,int,int]] = []
        wvec: List[float]             = []

        for layer, mat in enumerate((self.exc, self.inh, self.gap)):
            indptr, indices, data = mat.indptr, mat.indices, mat.data
            for i in range(mat.shape[0]):
                for p in range(indptr[i], indptr[i + 1]):
                    j   = int(indices[p])
                    ptr.append((layer, i, j))
                    if   layer == 0: wvec.append(+data[p])   # excitatory
                    elif layer == 1: wvec.append(-data[p])   # inhibitory (sign-)
                    else:            wvec.append(+data[p])   # gap
        self._edge_ptr = ptr
        self._edge_w   = np.asarray(wvec, dtype=np.float64)


    def _refresh_arrays(self):
        #for m in (self.gap, self.exc, self.inh):
            #m.eliminate_zeros()
        self._exc_d, self._exc_i, self._exc_p = map(np.asarray,
            (self.exc.data, self.exc.indices, self.exc.indptr))
        self._inh_d, self._inh_i, self._inh_p = map(np.asarray,
            (self.inh.data, self.inh.indices, self.inh.indptr))
        self._gap_d, self._gap_i, self._gap_p = map(np.asarray,
            (self.gap.data, self.gap.indices, self.gap.indptr))
        self._rebuild_edge_index()          # keep 1-D view in sync

    def _apply_weight_vector(self, vec: np.ndarray) -> None:
        """
        vec[k] must correspond 1-to-1 with wc[k] ordering:
            • >0  → excitatory (layer 0)
            • <0  → inhibitory (layer 1)
            • any sign  → gap (layer 2)  (magnitude used)
        """

        # clear all three matrices in-place
        self.exc.data[:] = 0.0
        self.inh.data[:] = 0.0
        self.gap.data[:] = 0.0

        for (layer, i, j), w in zip(self._edge_ptr, vec):
            if layer == 2:                      # gap: mirror
                self.gap[i, j] = self.gap[j, i] = abs(w)
            elif layer == 0:                    # excitatory slot
                if w < 0:
                    raise ValueError("Negative weight supplied for "
                                     "excitatory edge; sign mismatch.")
                self.exc[i, j] = w
            else:                               # inhibitory slot
                if w > 0:
                    raise ValueError("Positive weight supplied for "
                                     "inhibitory edge; sign mismatch.")
                self.inh[i, j] = -w             # store magnitude

        self._refresh_arrays()                  # rebuild views, _edge_w

    # ── 1-D edge interface ──────────────────────────────────
    def __getitem__(self, k: int):
        """Return (pre_name, post_name, weight) for edge #k."""
        if not isinstance(k, (int, np.integer)):
            raise TypeError("Index must be an int (1-D edge indexing only).")
        layer, i, j = self._edge_ptr[int(k)]
        if   layer == 0: w =  self.exc[i, j]
        elif layer == 1: w = -self.inh[i, j]          # inh stored positive
        else:            w =  self.gap[i, j]
        return (self.names[i], self.names[j], float(w))

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