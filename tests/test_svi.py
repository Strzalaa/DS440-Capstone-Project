"""Tests for SVI sentinel handling."""

import pandas as pd
import pytest

from src.svi import RPL_PCTILE_SENTINELS, mask_svi_percentile


def test_mask_replaces_sentinels_and_out_of_range():
    s = pd.Series([0.2, -999.0, 0.9, 1.5, -888.0])
    out = mask_svi_percentile(s)
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[3])
    assert pd.isna(out.iloc[4])
    assert out.iloc[0] == pytest.approx(0.2)
    assert out.iloc[2] == pytest.approx(0.9)


@pytest.mark.parametrize("v", list(RPL_PCTILE_SENTINELS))
def test_sentinels_na(v: int):
    out = mask_svi_percentile(pd.Series([float(v)]))
    assert pd.isna(out.iloc[0])
