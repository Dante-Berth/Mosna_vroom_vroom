#! /usr/bin/env python3
"""mosna.utils — small array/data helpers

Split out of the original monolithic ``mosna.mosna`` module.
"""
from mosna._common import *  # noqa: F401,F403


def to_numpy(data):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values.ravel()
    if gpu_clustering:
        # cupy is available
        if isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
    return data


def renormalize(data, mini, maxi):
    data = data - np.min(data)
    data = data / np.max(data)
    data = data * (maxi - mini)
    data = data + mini
    return data
