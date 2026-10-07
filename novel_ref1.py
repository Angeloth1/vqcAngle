"""
Novel-signature mask as defined in [1] and as used by the exp1/3/4 result files.

[1] (Section 7.5) counts as "novel" the KDDTest+ attack records whose attack type never
occurs in KDDTrain+_20Percent: 3,752 records. qids/data.py now counts the types absent from
the 80% training *partition* instead (3,759 records). That change was made after exp1/3/4
were run: every novel-recall value in their CSVs is a multiple of 1/3,752. New experiments
use this module so that all novel-recall numbers share one definition.
"""
from __future__ import annotations

import os
from pathlib import Path

from qids.data import _read

N_NOVEL_REF1 = 3752


def novel_mask_ref1(data_dir: str | None = None):
    here = Path(__file__).resolve().parent / "data"
    d = data_dir or os.environ.get("QIDS_DATA") or (str(here) if here.is_dir() else ".")
    tr = _read(os.path.join(d, "KDDTrain+_20Percent.txt"))
    te = _read(os.path.join(d, "KDDTest+.txt"))
    mask = ((~te["attack"].isin(set(tr["attack"]))) & (te["y"] == 1)).to_numpy()
    if int(mask.sum()) != N_NOVEL_REF1:
        raise ValueError(f"expected {N_NOVEL_REF1} novel records, got {int(mask.sum())}")
    return mask
