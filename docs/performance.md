# Performance, validation, and reproducibility supplement

## Reference problem

Except for the explicitly identified baseline comparison, performance values
in this document use `examples/synthetic_api_example/run_synthetic_rve.py`.

| Parameter | Value |
|---|---:|
| RVE | \(1.0\times1.0\times1.0\) mm |
| Voxel spacing | \(1\ \mu\mathrm{m}\) |
| Grid shape | \(1001\times1001\times1001\) |
| Voxels | 1,003,003,001 |
| Laser power | 370 W |
| Scan speed | 1.7 m s\(^{-1}\) |
| Hatch spacing | \(140\ \mu\mathrm{m}\) |
| Layer spacing | \(30\ \mu\mathrm{m}\) |
| Inter-layer rotation | \(67^\circ\) |
| Scan extension | 1 mm |
| Extra layers | 10 |
| Total padded vectors | 990 |
| Vectors overlapping the RVE | 816 |
| Spatial tiles | 169 |
| Mean candidates per tile | 561.78 |
| Random seed | 42 |

The generated 990-vector sequence is contiguous in time and ends at
1.74705882 s. In particular, all 44 inter-layer transitions begin at the end
time of the preceding layer's final vector.

The generated stochastic record contains 29,763 samples over
0.01750706 s. Tolerance-based compression retains 175 width modes, 159 depth
modes, and 89 height modes. Width reconstruction RMSE is
\(9.943344\times10^{-7}\) m. The upper and lower shape exponents are both one
for this example.

The packed table for active vectors contains 1,057,536 rows and occupies
12.10 MiB. Its reported conservative combined spectral/interpolation error
bound is \(0.250\ \mu\mathrm{m}\), one quarter of the voxel spacing.

A cached end-to-end execution reported 0.259 s for signature loading and
warmup, 1.310 s for spatial-index construction, spectral-table construction,
and melt-mask calculation, and 8.73 s total wall time including signal
generation, plotting, VTK output, and sparse morphology. Peak resident memory
was approximately 1.35 GiB. The optimized result contained 182,502 defect
voxels and 171 connected components after single-voxel removal.

## Benchmark environment

| Item | Value |
|---|---|
| Processor | AMD EPYC 7702 64-Core Processor |
| Available logical CPUs | 64 |
| Hardware threads per reported core | 1 |
| Memory | 125 GiB |
| Architecture | x86-64 with AVX2 and FMA |
| Operating system | Ubuntu 22.04 environment |
| Python | 3.10.12 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| Numba | 0.64.0 |
| llvmlite | 0.46.0 |
| Intel SVML | unavailable (`USING_SVML=False`) |

The host is virtualized with KVM. Results therefore characterize this
allocation, not every EPYC 7702 system.

## Timing protocol

Compilation and steady-state execution are reported separately. Numba
signatures were first compiled into a persistent cache:

```bash
export NUMBA_CACHE_DIR=/path/to/persistent/raptor-numba-cache
raptor-warm-cache --cache-dir "$NUMBA_CACHE_DIR"
```

For strong scaling, the complete path-vector preparation and spatial index were
constructed once. At each thread count, the optimized
`compute_melt_mask_grid` call was executed three times. The first observation
was treated as a settling run and the median of the final two was reported.
Spectral-table construction is included in each reported call. A SHA-256
checksum was computed at every thread count.

Tile-width measurements rebuild and time the spatial index for each width.
Tiles are candidate lookup bins; parallel work remains distributed over voxel
columns rather than over tiles. Weak scaling increases the \(x\)-\(y\) area in
proportion to thread count while holding \(z\), resolution, path density, and
melt-pool dimensions fixed.

The measurements are directly reproducible with:

```bash
cd examples/synthetic_api_example
python benchmark_scaling.py \
    --study strong \
    --threads 8 16 32 64 \
    --tile-widths-um 80 \
    --repeats 3 \
    --settling-runs 1 \
    --output /tmp/raptor_synthetic_scaling.csv
```

## Strong scaling

| Threads | Raw times (s) | Median retained time (s) | Throughput (Mvox s\(^{-1}\)) | Speedup from 8 | Parallel efficiency |
|---:|---|---:|---:|---:|---:|
| 8 | 7.000, 7.057, 7.085 | 7.071 | 141.9 | 1.000 | 100.0% |
| 16 | 3.927, 3.898, 3.901 | 3.900 | 257.2 | 1.813 | 90.7% |
| 32 | 2.164, 2.139, 2.143 | 2.141 | 468.4 | 3.302 | 82.6% |
| 64 | 1.252, 1.205, 1.216 | 1.210 | 828.6 | 5.842 | 73.0% |

All twelve measured fields produced the same checksum. Scaling remains useful
through 64 threads, although efficiency decreases as work per thread falls and
shared-memory bandwidth and scheduling costs become more prominent.

## Comparison with the main-branch implementation

The optimized working tree was compared with an isolated archive of main
commit `e6e7fb3794fc2ae977023d4464148ce676128369`. Both implementations used
the same regular grids, manually constructed horizontal paths, 32-mode
dimension spectra, deterministic phases, and 64 CPU threads. Times are medians
of recurring high-level porosity calls after JIT compilation and include path
preparation plus any index and spectral-table construction, but exclude VTK
and morphology.

| Voxels | Vectors | Main (s) | Optimized (s) | Speedup |
|---:|---:|---:|---:|---:|
| 1,030,301 | 90 | 0.2281 | 0.0112 | 20.4 |
| 8,120,601 | 272 | 1.6716 | 0.0556 | 30.1 |

The smaller optimized measurements have greater scheduler noise, so these
ratios characterize the tested host and workloads rather than a universal
speedup. The billion-voxel case was not executed on main: its resident
Float64 coordinate array alone would require approximately 22.4 GiB, with
additional full-size transient arrays during grid construction. No main-branch
billion-voxel timing is extrapolated from the smaller cases.

On the 8,120,601-voxel comparison, 0.3530% of phase codes and 0.00224% of
defect statuses differed. This is not a pure floating-point comparison because
it includes Newton-versus-implicit classification, spatial pruning, and
spectral lookup. The direct Float64 spectral comparison below isolates the
table approximation within the current geometry and is the preferred
numerical-accuracy assessment.

## Tile-width study

The complete billion-voxel field was evaluated at 64 threads for nine physical
tile widths. All 27 calls produced checksum
`b918f8d7aa1eeac00fb7f6a145e66115d728635704c37a4584c64da52d79aae4`.

| Width (µm) | Tiles | Mean candidates | Candidate entries | Index time (s) | Mask median (s) |
|---:|---:|---:|---:|---:|---:|
| 16 | 3,969 | 543.80 | 2,158,324 | 0.939 | 1.332 |
| 32 | 1,024 | 548.34 | 561,498 | 0.252 | 1.296 |
| 48 | 441 | 553.77 | 244,214 | 0.123 | 1.299 |
| 64 | 256 | 557.88 | 142,818 | 0.078 | 1.325 |
| 80 | 169 | 561.78 | 94,941 | 0.058 | 1.320 |
| 96 | 121 | 565.35 | 68,407 | 0.047 | 1.315 |
| 128 | 64 | 576.84 | 36,918 | 0.033 | 1.293 |
| 160 | 49 | 579.47 | 28,394 | 0.029 | 1.286 |
| 256 | 16 | 615.00 | 9,840 | 0.021 | 1.316 |

The single 64-thread sweep favored 160 µm by 2.6% over 80 µm. A second
cross-thread comparison found 160 µm 2.0% faster at 8 threads but 0.4%, 1.7%,
and 6.5% slower at 16, 32, and 64 threads. Because the host is virtualized and
the differences are small, the existing 80-µm automatic target is retained.
It is within 5% of the best width in the complete sweep and can now be
overridden through the public API.

The fact that 160 µm produced only 49 tiles yet remained competitive with 64
threads confirms that tiles are not a processor decomposition. Numba schedules
the 1,002,001 voxel columns independently.

## Weak scaling

The \(x\)-\(y\) area was proportional to thread count, yielding approximately
15.7 million voxels per thread. The fixed physical tile width was 80 µm.

| Threads | \(x\)-\(y\) scale | Voxels | Padded vectors | Median (s) | Weak efficiency |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.3536 | 126,151,025 | 360 | 0.857 | 100.0% |
| 16 | 0.5000 | 251,252,001 | 495 | 0.956 | 89.6% |
| 32 | 0.7071 | 501,765,264 | 720 | 1.083 | 79.1% |
| 64 | 1.0000 | 1,003,003,001 | 990 | 1.275 | 67.2% |

Mean candidates per tile increased from 282.0 to 561.78 as the physical area
grew. This global candidate-density growth, memory traffic, and scheduling
overhead explain the loss from ideal weak scaling more directly than the
number of tiles per worker.

## Spectral-error policy study

Ten thousand random active-vector/time evaluations were compared with
independent direct Float64 cosine evaluation. Full phase fields were also
compared with the strictest accepted budget, 0.15 voxel.

| Fraction | Table rows | Table MiB | Width max (µm) | Depth max (µm) | Height max (µm) | Changed labels | Defect-status changes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | rejected | — | — | — | — | — | — |
| 0.150 | 4,428,432 | 50.68 | 0.00136 | 0.00110 | 0.000289 | 0 | 0 |
| 0.200 | 1,454,928 | 16.65 | 0.01328 | 0.01083 | 0.00275 | 66,539 | 76 |
| 0.250 | 1,057,536 | 12.10 | 0.02329 | 0.01890 | 0.00451 | 132,682 | 149 |
| 0.500 | 577,728 | 6.61 | 0.09076 | 0.06408 | 0.01727 | 464,010 | 554 |

The 0.125 budget is rejected because the guarded Float32 node-error estimate
is 0.144 µm before interpolation. At 64 threads, median full-mask times for
fractions 0.15, 0.20, 0.25, and 0.50 were 1.473, 1.277, 1.257, and 1.281 s,
respectively. The default 0.25 fraction is retained: it was effectively the
fastest configuration while preserving substantially more phase agreement
than 0.50. These discrete phase differences are why the numerical policy is
public and must be reported with scientific results.

## Cold start and cache behavior

Before production-signature specialization, the observed first run spent
45.49 s compiling the general mask signature and another 5.40 s in the first
table-enabled mask call. The optimized-only precompiler requires approximately
10.2 s for the spectral table, production mask, and histogram signatures on
the reference host. Loading those entries in a subsequent process requires
approximately 0.29 s. Precompiling sparse morphology as well increases the
first warmup to 12.7 s; it adds only approximately 0.01 s to a cached warmup.

| Measurement | Earlier general path | Optimized-only path |
|---|---:|---:|
| Cold JIT warmup | 45.49 s | 10.24 s |
| Cold time through completed melt mask | 50.89 s | approximately 11.55 s |
| Cached signature load | — | approximately 0.29 s |

The first-run time through the melt mask was reduced by approximately 77.3%
(\(4.41\times\)). Source changes, Python/Numba upgrades, and CPU-feature
changes can invalidate the cache; the warm-cache command should be part of the
deployment or batch-job preparation step.

## Optimization decision record

### Retained changes

| Optimization | Rationale and observed effect |
|---|---|
| Lazy implicit grid | Eliminates a Float64 \(N\times3\) coordinate array. For the reference RVE this avoids approximately 22.4 GiB. |
| Signed 8-bit phase field | Stores the billion-voxel result in approximately 0.93 GiB rather than using a multi-byte integer field. |
| Padded-vector AABB filtering | Retains physically interacting out-of-domain paths while removing 174 of 990 vectors from the RVE calculation. |
| Adaptive 2-D spatial index | Restricts ordered vector tests to AABBs overlapping each \(x\)-\(y\) tile. |
| Column-parallel traversal | Makes \(z\) contiguous and hoists vector projection and spectral state out of the inner loop. |
| Conservative vertical interval | Restricts the inner loop to the only \(z\) values that can be melted or lie in the one-voxel boundary band. |
| Analytical/direct classification | The production kernel tests the implicit modified Lamé function and does not execute Newton iterations. |
| Mixed precision | Retains Float64 geometry and bounds while using guarded Float32 spectral tables. |
| Minimax-style cosine polynomial | Replaces library cosine in table construction after range reduction; the approximation is admitted only by the voxel-relative error guard. |
| Packed spectral histories | Reuses independent width/depth/height values across every column interacting with a vector. |
| Horizontal optimized-only kernel | Removes runtime branches and unused arguments for explicit grids, vertical paths, and direct spectral evaluation. |
| Sparse morphology | Avoids a second billion-element Boolean field when defects are sparse and requested properties are supported. |
| Lazy optional imports | Defers VTK, pandas, scikit-image, PyVista, and Matplotlib imports until requested. |
| Persistent Numba cache | Moves compilation out of simulation timing and reduces recurring startup to cache-load time. |

### Quantified kernel decisions

During optimization, the packed spectral table reduced the median kernel time
from 2.562 s for direct guarded spectral evaluation to 1.602 s, a 37.5% time
reduction (\(1.60\times\)). A later controlled comparison on the same
billion-voxel problem measured 1.411 s for the branch-containing general table
kernel and 1.231 s for the horizontal specialized kernel, a further 12.8%
reduction.

These development variants were intentionally removed from the final source.
They were measured before correction of the generated inter-layer timing
sequence and are retained only as historical optimization-decision data. The
current optimized-only timing is reproducible with `benchmark_scaling.py`;
historical alternatives are not presented as current reference measurements.

### Rejected alternatives

- A dense Clenshaw recurrence was numerically valid but approximately
  2.5--4 times slower for this workload. The recurrence is dependent in mode
  index and the retained FFT harmonics are sparse, while vector phases vary.
- Broad removal of forced Numba inlining reduced production compilation by
  about one second but slowed the billion-voxel kernel from 1.234 s to
  1.531 s (24%). Retaining only the per-voxel classifier inline still produced
  approximately 1.348 s (9% slower). Hot helper inlining was retained.
- Lower Numba optimization levels and generic CPU targets were not retained
  because recurring billion-voxel execution, rather than a one-time compile,
  is the primary objective.
- Numba's deprecated `pycc` path was not adopted. Persistent native-CPU cache
  entries preserve JIT specialization without introducing a second numerical
  implementation.
- Raw appended VTK encoding was faster and smaller but was not readable
  reliably by the targeted ParaView/VTK 9.1 reader for the billion-point file.
  Inline compressed binary output was retained for compatibility.

## Numerical validation

### Spectral error

For 10,000 representative dimension evaluations in the reference signal, the
observed maximum absolute errors relative to direct Float64 cosine evaluation
were approximately:

| Dimension | Maximum observed error |
|---|---:|
| Width | \(0.0240\ \mu\mathrm{m}\) |
| Depth | \(0.0206\ \mu\mathrm{m}\) |
| Height | \(0.00499\ \mu\mathrm{m}\) |

All are below the \(0.25\ \mu\mathrm{m}\) conservative budget used for the
1-\(\mu\)m grid.

### Phase sensitivity relative to direct spectra

The direct reference uses the same spatial candidates, culling operations,
Float64 geometry, classification, and ordered phase updates as the optimized
kernel. Only spectral lookup is replaced by direct Float64 cosine evaluation
at each accepted vector-column pair.

Comparing the retained table result with the direct spectral reference changed
140,443 labels among 1,003,003,001 voxels (0.014002%). Most changes exchanged
interior and boundary codes. Only 161 voxels changed defect/non-defect status:
121 changed from defect to boundary and 40 from boundary to defect. Both fields
contained 171 connected defect components after removal of single-voxel
objects.

The direct spectral reference had

```text
histogram: {0: 182583, 1: 938964859, 2: 61349292, 3: 2506267}
checksum:  40f6b65d966c0866c7f2a0fe0576dfcd0dcb9c94f15381f6007e928840c3e5cd
```

The optimized table result has

```text
histogram: {0: 182502, 1: 938963366, 2: 61350702, 3: 2506431}
checksum:  b918f8d7aa1eeac00fb7f6a145e66115d728635704c37a4584c64da52d79aae4
```

The discrepancy is substantially smaller than one voxel in geometric
resolution and is bounded by the explicit quarter-voxel spectral/interpolation
criterion. Nevertheless, both histograms and the selected numerical policy
should be reported because thresholded phase labels are discrete.

### Software verification

The focused optimized-only test suite covers:

- FFT mode selection and reconstruction tolerance;
- independent dimension mode and phase arrays;
- cosine and interpolation error bounds;
- public numerical/resource controls and tile-width invariance;
- optimized modified Lamé classification against radial signed distance;
- conservative vertical pruning;
- spatial-index retention of padded interacting vectors;
- contiguous scan timing across generated layer transitions;
- nonzero RVE origins and finite grid/path validation;
- finite and physically positive direct spectral inputs;
- fixed transverse-exponent enforcement;
- optimized-grid phase labels and orientation validation;
- deterministic seeded phases and checksums;
- sparse morphology against dense reference behavior;
- backward-compatible VTK round trips;
- CLI-relative input resolution and process exit status;
- comma- and whitespace-delimited melt-pool inputs; and
- malformed input and output-writer failure handling.

The suite passes with one environment warning caused by conflicting
Matplotlib installations and unavailable `Axes3D`; that warning does not
affect the numerical calculation.

## Reproduction commands

Install and precompile:

```bash
python -m pip install -e .
export NUMBA_CACHE_DIR=/path/to/persistent/raptor-numba-cache
raptor-warm-cache --cache-dir "$NUMBA_CACHE_DIR"
```

Run the reference calculation:

```bash
cd examples/synthetic_api_example
python run_synthetic_rve.py
```

Run verification:

```bash
cd /path/to/Raptor
pytest -q
```

For a timing study, retain the same process affinity, set the desired Numba
thread count before execution, warm the cache, exclude VTK and plotting from
the kernel interval, and report all individual timing samples rather than only
the minimum.
