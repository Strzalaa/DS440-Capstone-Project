"""CDC Social Vulnerability Index (SVI) value handling.

SVI tract files use sentinel values such as -999 (and sometimes -888) for
missing or unreliable estimates. RPL_* fields are overall and theme percentile
ranks and should lie in [0, 1].
"""

from __future__ import annotations

import pandas as pd

# Documented CDC SVI sentinels for missing / insufficient data in published CSVs.
RPL_PCTILE_SENTINELS: tuple[int, ...] = (-999, -888)


def mask_svi_percentile(s: pd.Series) -> pd.Series:
    """Return *s* with invalid RPL percentile values set to NA.

    Parameters
    ----------
    s : pd.Series
        An RPL_THEMES / RPL_THEME* / svi_overall column.
    """
    out = pd.to_numeric(s, errors="coerce")
    out = out.mask(out.isin(RPL_PCTILE_SENTINELS))
    out = out.where(out.isna() | out.between(0.0, 1.0, inclusive="both"))
    return out
