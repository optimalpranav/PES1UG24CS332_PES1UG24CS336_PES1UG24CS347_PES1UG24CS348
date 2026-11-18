"""
Contract for the HMM member:  hmm/viterbi.py must expose
    decode(cands: pd.DataFrame, trip: dict) -> pd.Series   (index = obs_idx, value = arc_id)
cands has columns obs_idx, arc_id, timestamp, prob + FEATURE_COLS.
trip has track/nodes/arcs only; trip["route"] is None (never use it).
"""
try:
    from hmm.viterbi import decode as viterbi_decode
except ImportError:
    viterbi_decode = None


def argmax_decode(cands):
    best = cands.loc[cands.groupby("obs_idx")["prob"].idxmax()]
    return best.set_index("obs_idx")["arc_id"].sort_index()


def decode(cands, trip):
    if viterbi_decode is None:
        return argmax_decode(cands), "argmax"
    return viterbi_decode(cands, trip).sort_index(), "viterbi"