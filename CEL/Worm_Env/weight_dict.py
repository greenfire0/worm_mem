from __future__ import annotations
#from util.read_from_xls import combine_neuron_data,get_all_neuron_names
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import scipy.sparse as sp
import numpy.typing as npt
muscles = ['MVU', 'MVL', 'MDL', 'MVR', 'MDR']
muscleList = ['MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23', 'MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDR21', 'MDR22', 'MDR23', 'MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVL21', 'MVR22', 'MVR23']
#some face muscles not included

mLeft = ['MDL07', 'MDL08', 'MDL09', 'MDL10', 'MDL11', 'MDL12', 'MDL13', 'MDL14', 'MDL15', 'MDL16', 'MDL17', 'MDL18', 'MDL19', 'MDL20', 'MDL21', 'MDL22', 'MDL23', 'MVL07', 'MVL08', 'MVL09', 'MVL10', 'MVL11', 'MVL12', 'MVL13', 'MVL14', 'MVL15', 'MVL16', 'MVL17', 'MVL18', 'MVL19', 'MVL20', 'MVL21', 'MVL22', 'MVL23']
mRight = ['MDR07', 'MDR08', 'MDR09', 'MDR10', 'MDR11', 'MDR12', 'MDR13', 'MDR14', 'MDR15', 'MDR16', 'MDR17', 'MDR18', 'MDR19', 'MDR20', 'MDL21', 'MDR22', 'MDR23', 'MVR07', 'MVR08', 'MVR09', 'MVR10', 'MVR11', 'MVR12', 'MVR13', 'MVR14', 'MVR15', 'MVR16', 'MVR17', 'MVR18', 'MVR19', 'MVR20', 'MVL21', 'MVR22', 'MVR23']
file_path = 'CElegansNeuronTables.xlsx'
#dict = combine_neuron_data(file_path)
#all_neuron_names=get_all_neuron_names(dict)
#print(all_neuron_names)
# Used to accumulate muscle weighted values in body muscles 07-23 = worm locomotion

from collections import OrderedDict
from pathlib import Path
import numpy as np




import re
def _norm_nt(raw: str) -> str:
    """
    Returns a canonical transmitter label.

    * Lower-cases, strips spaces/underscores.
    * Splits co-transmission strings on ',', ';', '/', or whitespace.
    * Sorts sub-parts so 'serotonin acetylcholine' == 'acetylcholine_serotonin'.
    * Normalises aliases & typos.
    """
    if pd.isna(raw):
        return 'unknown'

    s = raw.lower().replace('_', ' ').replace('/', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    aliases = {
        'ach': 'acetylcholine',
        'frmfemide': 'fmrfamide',
        'gapjunction': 'generic_gj',
        'generic gj': 'generic_gj',
    }
    s = aliases.get(s, s)

    parts = re.split(r'[;, ]', s)
    parts = [aliases.get(p, p) for p in parts if p]
    return ','.join(sorted(parts)) if len(parts) > 1 else parts[0]

# ──────────────────────────────────────────────────────────────
# 1.  read both sheets → tidy DF  (add NT normalisation)
# ──────────────────────────────────────────────────────────────
def load_raw_xlsx(path: str | Path) -> pd.DataFrame:
    c = (pd.read_excel(path, sheet_name="Connectome")
           .rename(columns=str.strip)
           .assign(Type=lambda d: d.Type.str.lower(),
                   NT=lambda d: d.Neurotransmitter.apply(_norm_nt))
           [['Origin','Target','Type','NT','Number of Connections']]
           .rename(columns={'Number of Connections':'Number'}))

    m = (pd.read_excel(path, sheet_name="NeuronsToMuscle")
           .rename(columns=str.strip)
           .rename(columns={'Neuron':'Origin','Muscle':'Target',
                            'Number of Connections':'Number'})
           .assign(Type='neuromuscular',
                   NT=lambda d: d.get('Neurotransmitter','acetylcholine')
                                 .apply(_norm_nt))[c.columns])

    df = pd.concat([c, m], ignore_index=True)
    df['Number'] = df['Number'].astype(int)
    return df

# ──────────────────────────────────────────────────────────────
# 2.  sparse layers  (unchanged logic, but return nt_counts)
# ──────────────────────────────────────────────────────────────
def df_to_sparse_layers(df: pd.DataFrame
        ) -> Tuple[List[str], Dict[str,int], sp.csr_matrix, sp.csr_matrix,
                   sp.csr_matrix, Dict[str,int]]:

    neurons  = pd.Index(sorted(set(df.Origin)|set(df.Target)))
    name2idx = {n:i for i,n in enumerate(neurons)}
    src      = df.Origin.map(name2idx).to_numpy(np.int32)
    dst      = df.Target.map(name2idx).to_numpy(np.int32)
    w        = df['Number'].to_numpy(np.float64)

    # ---------- sign & class masks ---------------------------------
    is_gaba  = df.NT.eq('gaba')
    w[is_gaba] *= -1                    # store GABA as negative

    is_gap   = df.Type.eq('gapjunction')
    is_chem  = ~is_gap                  # anything not a gap is chemical
    is_inh   = w < 0                    # now signed
    is_exc   = is_chem & ~is_inh        # *all* non-GABA chems

    # ---------- build sparse layers --------------------------------
    chem_exc = sp.coo_matrix(( np.abs(w[is_exc]),
                               (src[is_exc], dst[is_exc]) ),
                             shape=(len(neurons),)*2).tocsr()
    chem_inh = sp.coo_matrix(( np.abs(w[is_inh]),
                               (src[is_inh], dst[is_inh]) ),
                             shape=(len(neurons),)*2).tocsr()
    gap      = sp.coo_matrix(( np.abs(w[is_gap]),
                               (src[is_gap], dst[is_gap]) ),
                             shape=(len(neurons),)*2).tocsr()
    gap      = (gap + gap.T) * 0.5

    nt_counts = df.NT.value_counts().sort_index().to_dict()
    return neurons.tolist(), name2idx, chem_exc, chem_inh, gap, nt_counts


# ──────────────────────────────────────────────────────────────
# 3.  refresh_npz  (one-liner update)
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# 3.  (unchanged)  Save bundle to disk
# ──────────────────────────────────────────────────────────────────────
def export_npz(neurons, exc, inh, gap, nt_counts: dict[str, int],
               out_path: str | Path = "connectome_sparse.npz"):
    np.savez_compressed(
        out_path,
        neurons=np.array(neurons),
        exc_data=exc.data, exc_indices=exc.indices, exc_indptr=exc.indptr,
        inh_data=inh.data, inh_indices=inh.indices, inh_indptr=inh.indptr,
        gap_data=gap.data, gap_indices=gap.indices, gap_indptr=gap.indptr,
        shape=np.array(exc.shape, dtype=np.int32),
        nt_keys=np.array(list(nt_counts.keys())),
        nt_vals=np.array(list(nt_counts.values()), dtype=np.int32),
    )
    print(f"saved ➜ {out_path}")


def refresh_npz():
    df = load_raw_xlsx("CElegansNeuronTables.xlsx")
    neurons, n2i, exc, inh, gap, nt_counts = df_to_sparse_layers(df)
    export_npz(neurons, exc, inh, gap, nt_counts)