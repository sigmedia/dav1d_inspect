from __future__ import annotations

from itertools import islice
from pathlib import Path

import numpy as np

from dav1d_inspect import iter_frames, iter_rgb_frames


IVF = Path(__file__).with_name("0018.ivf")


def test_rgb_shim_matches_callback_and_samples() -> None:
    native = list(iter_rgb_frames(IVF, n_threads=2, max_frames=2))
    callback = list(islice(iter_frames(IVF, n_threads=2, want_rgb=True), 2))

    assert len(native) == len(callback) == 2
    for fast, slow in zip(native, callback, strict=True):
        assert fast["rgb"].shape == (1080, 1920, 3)
        assert fast["rgb"].dtype == np.uint8
        difference = np.abs(
            fast["rgb"].astype(np.int16) - slow["rgb"].astype(np.int16)
        )
        assert difference.max() <= 1

    sampled = list(
        iter_rgb_frames(
            IVF,
            n_threads=2,
            step=3,
            max_frames=3,
            target_size=(64, 48),
            process_motion=True,
            linear_interpolation=True,
            normalize_flow=True,
            nan_to_num=True,
        )
    )
    assert [frame["decode_seq"] for frame in sampled] == [0, 3, 6]
    assert len(sampled) < len(list(iter_rgb_frames(IVF, step=3, max_frames=4)))
    for frame in sampled:
        assert frame["rgb"].shape == (48, 64, 3)
        assert frame["rgb"].dtype == np.uint8
        assert frame["motion_field"].shape == (48, 64, 2)
        assert frame["motion_field"].dtype == np.float32
        assert np.isfinite(frame["motion_field"]).all()
        assert np.abs(frame["motion_field"]).max() <= 1.0
