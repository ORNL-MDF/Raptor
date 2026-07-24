"""Public Raptor API with lazy imports for optional heavy dependencies."""

from importlib import import_module

__all__ = [
    "compute_spectral_components",
    "compute_porosity",
    "compute_phase_histogram",
    "write_vtk",
    "read_scan_path",
    "read_data",
    "PathVector",
    "MeltPool",
]

_EXPORTS = {
    "compute_spectral_components": (
        "raptor.api",
        "compute_spectral_components",
    ),
    "compute_porosity": ("raptor.api", "compute_porosity"),
    "compute_phase_histogram": ("raptor.api", "compute_phase_histogram"),
    "write_vtk": ("raptor.api", "write_vtk"),
    "read_scan_path": ("raptor.io", "read_scan_path"),
    "read_data": ("raptor.io", "read_data"),
    "PathVector": ("raptor.structures", "PathVector"),
    "MeltPool": ("raptor.structures", "MeltPool"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'raptor' has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
