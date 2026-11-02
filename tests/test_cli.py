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


def _write_case(tmp_path, melt_pool_yaml, extra_parameters=""):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    np.savetxt(
        case_dir / "dimension.txt",
        np.array([[0.0, 100.0e-6], [1.0, 100.0e-6]]),
    )
    np.savetxt(
        case_dir / "spectrum.csv",
        np.array([[100.0e-6, 0.0], [5.0e-6, 1.0e3]]),
        delimiter=",",
    )
    (case_dir / "scan.txt").write_text(
        "Mode X Y Z Power Parameter\n"
        "1 0 0 0 0 0\n"
        "0 1e-5 0 0 100 1\n"
        "1 0 0 3e-5 0 0\n"
        "0 1e-5 0 3e-5 100 1\n"
    )
    (case_dir / "input.yaml").write_text(
        f"""
scan_paths: [scan.txt]
parameters:
  layer_height: 3.0e-5
  voxel_resolution: 1.0e-5
  enable_random_segment_phase: true
  random_seed: 3
{extra_parameters}
melt_pool_data:
{melt_pool_yaml}
rve:
  min_point: [0.0, -1.0e-5, -1.0e-5]
  max_point: [1.0e-5, 1.0e-5, 1.0e-5]
output: {{}}
""".lstrip()
    )
    return case_dir


SPECTRAL_MELT_POOL = """
  width: {type: spectral_components, file_name: spectrum.csv, scale: 1.0,
          mean: 1.2e-4}
  depth: {type: spectral_components, file_name: spectrum.csv, scale: 0.8,
          shape: 1.0, mean: 1.2e-4}
  height: {type: spectral_components, file_name: spectrum.csv, scale: 0.2,
           shape: 1.0, mean: 1.2e-4}
""".strip(
    "\n"
)


def _run_cli(case_dir, monkeypatch):
    observed = {}

    def compute_porosity(grid, vectors, melt_pool, **options):
        observed.update(vectors=vectors, melt_pool=melt_pool)
        return np.ones(grid.shape, dtype=np.int8)

    monkeypatch.setattr(cli, "compute_porosity", compute_porosity)
    monkeypatch.setattr(sys, "argv", ["raptor", str(case_dir / "input.yaml")])
    return cli.main(), observed


def test_cli_accepts_spectral_components(tmp_path, monkeypatch):
    case_dir = _write_case(tmp_path, SPECTRAL_MELT_POOL)
    status, observed = _run_cli(case_dir, monkeypatch)

    assert status == 0
    melt_pool = observed["melt_pool"]
    assert melt_pool.width_mean == pytest.approx(1.2e-4)
    assert melt_pool.depth_mean == pytest.approx(0.8 * 1.2e-4)
    assert melt_pool.depth_shape_factor == 1.0


def test_cli_rejects_mixed_melt_pool_types(tmp_path, monkeypatch, capsys):
    mixed = SPECTRAL_MELT_POOL.replace(
        "width: {type: spectral_components, file_name: spectrum.csv,",
        "width: {type: time_series, file_name: dimension.txt, nmodes: 1,",
    )
    case_dir = _write_case(tmp_path, mixed)
    status, _ = _run_cli(case_dir, monkeypatch)

    assert status == 1
    assert "same type" in capsys.readouterr().out


def test_cli_powder_consolidation_stretches_scan_path_height(
    tmp_path, monkeypatch
):
    case_dir = _write_case(
        tmp_path, SPECTRAL_MELT_POOL, "  powder_consolidation: 0.5"
    )
    status, observed = _run_cli(case_dir, monkeypatch)

    assert status == 0
    heights = sorted(v.start_point[2] for v in observed["vectors"])
    assert heights == pytest.approx([0.0, 6.0e-5])


def test_cli_rejects_invalid_powder_consolidation(
    tmp_path, monkeypatch, capsys
):
    case_dir = _write_case(
        tmp_path, SPECTRAL_MELT_POOL, "  powder_consolidation: 2.0"
    )
    status, _ = _run_cli(case_dir, monkeypatch)

    assert status == 1
    assert "powder_consolidation" in capsys.readouterr().out
