# =============================================================================
# Copyright (c) 2025 Oak Ridge National Laboratory
#
# All rights reserved.
#
# This file is part of Raptor.
#
# For details, see the top-level LICENSE file at:
# https://github.com/ORNL-MDF/Raptor/LICENSE
# =============================================================================
import numpy as np
import pytest

from raptor.api import compute_spectral_components
from raptor.io import read_data, read_scan_path


def test_read_data_preserves_uniform_timestamp_precision(tmp_path):
    time = np.arange(30_000, dtype=np.float64) * (1.0 / 6_137_000.0)
    values = 1.0e-4 + 1.0e-5 * np.cos(2.0 * np.pi * 100.0 * time)
    input_path = tmp_path / "melt_pool_data.csv"
    np.savetxt(input_path, np.column_stack([time, values]), delimiter=",")

    data = read_data(input_path)
    spectral_components = compute_spectral_components(data, n_modes=2)

    assert data.dtype == np.float64
    assert spectral_components.shape == (2, 3)


def test_read_data_accepts_whitespace_delimited_text(tmp_path):
    input_path = tmp_path / "melt_pool_data.txt"
    np.savetxt(input_path, np.array([[0.0, 1.0], [1.0, 2.0]]))

    np.testing.assert_array_equal(
        read_data(input_path),
        [[0.0, 1.0], [1.0, 2.0]],
    )


def test_header_only_scan_path_is_empty(tmp_path):
    input_path = tmp_path / "scan_path.txt"
    input_path.write_text("Mode X Y Z Power Parameter\n")

    assert read_scan_path(input_path) == []


def test_malformed_scan_path_row_is_rejected(tmp_path):
    input_path = tmp_path / "scan_path.txt"
    input_path.write_text(
        "Mode X Y Z Power Parameter\n" "1 0 0 0 0 0\n" "0 1e-5 0 0 100\n"
    )

    with pytest.raises(ValueError, match="row 3"):
        read_scan_path(input_path)
