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
import sys

import numpy as np
import pytest

from raptor import cli


def test_cli_resolves_inputs_relative_to_config(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    np.savetxt(
        case_dir / "dimension.txt",
        np.array([[0.0, 100.0e-6], [1.0, 100.0e-6]]),
    )
    (case_dir / "scan.txt").write_text(
        "Mode X Y Z Power Parameter\n" "1 0 0 0 0 0\n" "0 1e-5 0 0 100 1\n"
    )
    (case_dir / "input.yaml").write_text(
        """
scan_paths: [scan.txt]
parameters:
  layer_height: 1.0e-5
  voxel_resolution: 1.0e-5
  enable_random_segment_phase: false
  memory_limit_mb: 512
melt_pool_data:
  width: {type: time_series, file_name: dimension.txt, nmodes: 1, scale: 1.0}
  depth:
    type: time_series
    file_name: dimension.txt
    nmodes: 1
    scale: 1.0
    shape: 2.0
  height:
    type: time_series
    file_name: dimension.txt
    nmodes: 1
    scale: 1.0
    shape: 2.0
rve:
  min_point: [0.0, -1.0e-5, -1.0e-5]
  max_point: [1.0e-5, 1.0e-5, 1.0e-5]
output: {}
""".lstrip()
    )
    monkeypatch.chdir(tmp_path)
    observed_options = {}

    def compute_porosity(grid, vectors, melt_pool, **options):
        observed_options.update(options)
        return np.ones(grid.shape, dtype=np.int8)

    monkeypatch.setattr(
        cli,
        "compute_porosity",
        compute_porosity,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["raptor", str(case_dir / "input.yaml")],
    )

    assert cli.main() == 0
    assert observed_options["memory_limit_mb"] == 512


def test_console_entry_point_propagates_failure_status(monkeypatch):
    monkeypatch.setattr(cli, "main", lambda: 1)

    with pytest.raises(SystemExit) as exit_info:
        cli.run()

    assert exit_info.value.code == 1


def test_cli_rejects_empty_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "input.yaml"
    config_path.write_text("")
    monkeypatch.setattr(sys, "argv", ["raptor", str(config_path)])

    assert cli.main() == 1
