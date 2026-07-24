# Raptor methods and reproducibility documentation

This directory describes the optimized CPU implementation of Raptor in a form
intended for direct adaptation into a journal manuscript and its computational
supplement.

- [methods.md](methods.md) presents the physical representation, governing
  equations, numerical approximations, voxel classification algorithm, and
  implementation constraints.
- [performance.md](performance.md) records the synthetic benchmark,
  optimization decisions, numerical validation, strong scaling, and commands
  needed to reproduce the reported results.

The documentation describes the optimized-only implementation on this branch
as of 2026-07-24. Before submission, associate the manuscript with an archived
release, commit identifier, and software DOI. Performance values should be
reported with the hardware and software environment given in
[performance.md](performance.md); they should not be generalized to other
systems without new measurements.

## Reproducing the documented calculation

From the repository root:

```bash
python -m pip install -e .

export NUMBA_CACHE_DIR=/path/to/persistent/raptor-numba-cache
raptor-warm-cache --cache-dir "$NUMBA_CACHE_DIR"

cd examples/synthetic_api_example
python run_synthetic_rve.py
```

The reference calculation is considered reproduced when the phase histogram
and SHA-256 checksum are

```text
{0: 182502, 1: 938963366, 2: 61350702, 3: 2506431}
b918f8d7aa1eeac00fb7f6a145e66115d728635704c37a4584c64da52d79aae4
```

Timing should be collected after the Numba cache has been warmed. Cold
compilation and steady-state execution are separate measurements.

The combined performance and accuracy study is reproduced with:

```bash
python benchmark_scaling.py \
    --study strong \
    --threads 8 16 32 64 \
    --tile-widths-um 16 32 48 64 80 96 128 160 256

python benchmark_scaling.py \
    --study weak \
    --threads 8 16 32 64

python benchmark_scaling.py \
    --study accuracy \
    --threads 64 \
    --spectral-error-fractions 0.15 0.20 0.25 0.50 \
    --accuracy-samples 10000 \
    --phase-accuracy
```

## Recommended manuscript reporting checklist

A publication using Raptor should report:

1. Raptor release, commit, and archive identifier.
2. Python, NumPy, SciPy, Numba, and LLVM/llvmlite versions.
3. CPU model, available cores, selected Numba thread count, and memory.
4. RVE bounds, voxel spacing, and resulting integer grid shape.
5. Scan speed, hatch spacing, layer spacing, inter-layer rotation, scan
   extension, and number of padded layers.
6. The source and preprocessing of width, depth, and height histories.
7. The number of retained modes for each dimension and the reconstruction
   tolerance or error.
8. Shape exponents for the upper and lower melt-pool surfaces.
9. Random-phase policy and random seed.
10. Spectral error fraction, tile width, spectral-table memory limit, resolved
    tile size, table sample count, and reported error bound.
11. Whether timing includes compilation, index and table construction,
    morphology, and VTK output.
12. Phase histogram, deterministic checksum, and requested morphology fields.
