# Journal-ready description of the Raptor method

## Scope

Raptor predicts lack-of-fusion porosity in a voxelized representative volume
element (RVE) by sweeping a time-dependent melt-pool cross-section along an
ordered laser scan path. The optimized CPU implementation represents width,
depth, and height as separate spectral histories; constructs a conservative
spatial index over padded scan vectors; and classifies the RVE in parallel by
two-dimensional voxel columns. The algorithm does not materialize the
three-coordinate array for the full RVE.

The current production kernel is specialized for horizontal scan vectors whose
local build direction coincides with the global \(z\) axis. This specialization
is deliberate: it permits the spectral state and transverse geometry to be
evaluated once per candidate vector and \((x,y)\) column, followed by a short,
contiguous traversal in \(z\). Non-horizontal vectors are rejected with an
explicit error.

## Input representation

### Voxel grid

For lower and upper RVE bounds \(\mathbf{x}_{\min}\) and
\(\mathbf{x}_{\max}\), and isotropic voxel spacing \(\Delta\), the grid has

$$
N_q =
\operatorname{len}\left[
q_{\min}, q_{\min}+\Delta,\ldots,
q_{\max}+\frac{\Delta}{2}
\right],
\qquad q\in\{x,y,z\},
$$

and \(N=N_xN_yN_z\) points. Coordinates are reconstructed from the origin and
integer indices only when needed:

$$
\mathbf{x}_{ijk} =
\mathbf{x}_{\min} + \Delta(i,j,k).
$$

The phase field is stored in C order with \(z\) contiguous. A coordinate array
of shape \(N\times3\) is therefore absent from the production calculation.
Backward-compatible coordinate materialization remains available through the
`Grid.voxels` property but is not used by the optimized kernel.

### Scan path and padding

A base hatch layer is generated at the RVE minimum \(z\) coordinate on a
rectangular domain extended by `scan_extension` on every in-plane side. Layer
\(l\) is rotated through

$$
\theta_l=l\theta_{\mathrm{rot}}
$$

about the in-plane center of the RVE. Additional layers may be placed above the
RVE. Retaining these out-of-domain vector portions and layers is necessary
because their melt-pool envelopes can intersect the RVE.

For scan vector \(j\), the start and end points are
\(\mathbf{s}_j\) and \(\mathbf{e}_j\), the displacement is
\(\mathbf{d}_j=\mathbf{e}_j-\mathbf{s}_j\), and the duration is

$$
T_j=\frac{\lVert\mathbf{d}_j\rVert}{v},
$$

where \(v\) is scan speed. The local axial direction is
\(\mathbf{e}_{1,j}=\mathbf{d}_j/\lVert\mathbf{d}_j\rVert\), the in-plane
transverse direction is
\(\mathbf{e}_{0,j}=(-e_{1y},e_{1x},0)\), and
\(\mathbf{e}_{2,j}=(0,0,1)\).

Generated vectors are contiguous in time: \(t_{0,0}=0\) and
\(t_{0,j}=t_{0,j-1}+T_{j-1}\). Consequently, a new layer begins only after the
last vector in the preceding layer has completed.

Each vector receives an axis-aligned bounding box (AABB) padded by half the
maximum width in \(x\) and \(y\), by the maximum depth below the vector, and by
the maximum height above it. Each maximum is conservatively bounded by the sum
of the absolute spectral amplitudes, so signed coefficients cannot shrink the
interaction envelope. An oriented in-plane test additionally uses
half-width \(W_{\max}/2\) and half-vector-length
\(\lVert\mathbf{d}_{j,xy}\rVert/2\).

## Melt-pool histories

### Optional stochastic signal model

The synthetic example generates a Gaussian width history from one or more
physical correlation scales. Sampling frequency is coupled to the voxel grid:

$$
f_s=\frac{v}{\Delta}.
$$

A physical length scale \(\ell_p\) is converted to center frequency
\(f_p=v/\ell_p\). A fourth-order Butterworth band-pass filter spanning
\([0.5f_p,1.5f_p]\) is applied to a shared white-noise realization. Component
standard deviations are normalized to their requested contributions, followed
by a common covariance correction that enforces the requested standard
deviation of their sum. The prescribed mean is added last.

The synthetic-record duration is selected deterministically. Raptor estimates
the autocorrelation implied by the configured filters and approximates the
effective sample count for a Gaussian variance estimate as

$$
N_{\mathrm{eff}} =
\frac{N}{
1+2\sum_{\tau=1}^{N-1}
\left(1-\frac{\tau}{N}\right)\rho(\tau)^2
}.
$$

The shortest sample count satisfying the configured chi-squared variance
confidence interval is found by expansion followed by binary search. This
procedure controls the statistical precision of the generated record; it is
distinct from the subsequent Fourier reconstruction tolerance.

### Independent spectral dimensions

Width \(W\), depth \(D\), and height \(H\) are represented independently:

$$
Q_{d,j}(\tau)=
\sum_{k=0}^{M_d-1}
A_{d,k}\cos\left(
2\pi f_{d,k}\tau+\phi_{d,j,k}
+2\pi f_{d,k}t_{0,j}
\right),
\qquad
d\in\{W,D,H\},
$$

where \(\tau\in[0,T_j]\) is local vector time and \(t_{0,j}\) is the vector
start time. Each dimension may contain a different number of modes \(M_d\);
mode arrays are not padded to a common length. This permits width, depth, and
height spectra from independent high-fidelity calculations to be supplied
without changing the kernel.

Input arrays, scales, and shape factors are validated before vector culling.
Spectra must be finite, begin with a positive zero-frequency dimension, and
use positive scale and shape factors. Signed coefficients are normalized to an
equivalent non-negative-amplitude/shifted-phase representation.
The transverse width exponent is required to equal two because that exponent
is fixed in the production kernel.
After table construction, every evaluated width, depth, and height must be
finite and positive; otherwise execution stops with a descriptive error.

When phases are supplied with spectral input and randomization is disabled,
each dimension retains its input phases. When random segment phases are
enabled, the DC phase is zero and a reproducible random phase sequence is
generated for each vector. Corresponding mode indices share that random
sequence across dimensions and each dimension truncates it to its own mode
count. Thus the spectral evaluations and amplitudes remain dimension-specific,
while the optional stochastic phase assignment can introduce correlation
between corresponding modes. All random phases are controlled by an explicit
seed.

### Fourier compression

Uniformly sampled input is transformed with a real FFT. Let \(X_k\) be the FFT
coefficient and \(N_s\) the number of samples. Positive-frequency cosine
amplitudes are

$$
A_k=\frac{w_k|X_k|}{N_s},
$$

where \(w_k=2\) for an interior real-FFT bin and \(w_k=1\) for DC and the
Nyquist bin. The phase is corrected for a nonzero initial time:

$$
\phi_k=\arg(X_k)-2\pi f_kt_{\min}.
$$

For tolerance-based compression, non-DC bins are ranked by Parseval energy
\(w_k|X_k|^2\). The smallest retained set satisfying

$$
\sum_{k\in\mathrm{discarded}}w_k|X_k|^2
\leq
(\varepsilon_{\mathrm{RMSE}}N_s)^2
$$

is selected and then returned in frequency order. If a mode-count cap is also
specified, the highest-energy modes are retained and Raptor warns when the
requested tolerance cannot be met.

## Spectral acceleration and numerical error control

### Float32 cosine approximation

Spectral tables are generated in single precision, while geometric
coordinates, projection operations, bounds, and error estimates remain in
double precision. A cosine argument is reduced to \([-\pi,\pi]\), reflected
onto \([-\pi/2,\pi/2]\), and evaluated by a degree-10 even polynomial in
\(r\), equivalently a degree-5 polynomial in \(z=r^2\):

$$
\begin{aligned}
P(z)=1+z(&-4.9999997\times10^{-1}
+z(4.1666638\times10^{-2}\\
&+z(-1.3888378\times10^{-3}
+z(2.4760495\times10^{-5}
-2.6051615\times10^{-7}z)))).
\end{aligned}
$$

The reflected sign is applied after evaluation. Horner evaluation and Numba
`fastmath` permit fused arithmetic where supported. The coefficients are the
degree-10 minimax coefficients used by Microsoft DirectXMath's
[`XMScalarCos`](https://github.com/microsoft/DirectXMath/blob/main/Inc/DirectXMathMisc.inl).

### Conservative approximation bound

Raptor accepts the Float32 spectral path only when a guarded, dimension-wise
estimate is no greater than a requested fraction \(\alpha\) of one voxel.
The public default is \(\alpha=0.25\). With Float32 machine epsilon \(u_{32}\),
maximum duration \(T_{\max}\), and \(M_d\) modes,

$$
\theta_{\max,d}
=2\pi T_{\max}\max_k|f_{d,k}|+2\pi,
$$

$$
\epsilon_{\cos,d}
=3\times10^{-7}
+16u_{32}(\theta_{\max,d}+1),
$$

$$
\epsilon_{\mathrm{acc},d}=2M_du_{32},
$$

and the absolute spectral bound is

$$
E_d =
\sum_k |A_{d,k}|
\min\left[
2,\epsilon_{\cos,d}+\epsilon_{\mathrm{acc},d}
\right].
$$

Execution stops with a descriptive error if

$$
\max_d E_d > \alpha\Delta.
$$

The \(3\times10^{-7}\) term is an empirical baseline allowance for the
reduced polynomial. The argument-dependent term covers Float32 conversion and
range-reduction error, and the accumulation term grows with mode count. This
is a guarded engineering bound rather than a formal all-input proof.

### Packed spectral lookup table

For dimension \(d\), an exact curvature bound for the cosine expansion is

$$
C_d=\sum_k |A_{d,k}|(2\pi f_{d,k})^2.
$$

Linear interpolation on an interval of duration \(h\) has error bounded by
\(C_dh^2/8\). The global table interval therefore satisfies

$$
h \leq
\min_d
\sqrt{\frac{8(\alpha\Delta-E_d)}{C_d}},
$$

with constant dimensions requiring only their endpoints. Vector \(j\) receives

$$
N_j=\max\left[
2,\left\lceil\frac{|T_j|}{h}\right\rceil+1
\right]
$$

samples. The three dimension histories are evaluated independently and stored
as packed rows `[width, depth, height]`; an offset array locates the ragged
table for each vector. The default production limit is 256 MiB and can be
changed through `max_spectral_table_bytes`. Exceeding the selected limit
raises an error rather than silently selecting a less accurate algorithm.

At a projected vector fraction \(u\), the kernel performs one linear
interpolation between adjacent packed rows. This replaces hundreds of cosine
evaluations at every interacting voxel column with six table loads and three
linear interpolations.

## Melt-pool geometry and voxel classification

### Modified Lamé cross-section

At local transverse coordinate \(y\) and build-direction coordinate \(z\), the
dynamic cross-section is

$$
F(y,z)=
\left(\frac{y}{W/2}\right)^2
+
\left(\frac{|z|}{B(z)}\right)^{n(z)},
$$

where

$$
B(z)=
\begin{cases}
H, & z\geq0,\\
D, & z<0,
\end{cases}
\qquad
n(z)=
\begin{cases}
n_H, & z\geq0,\\
n_D, & z<0.
\end{cases}
$$

The nominal melt region satisfies \(F<1\). The transverse exponent is fixed at
two; the upper and lower exponents control parabolic, elliptical, or more
general modified Lamé surfaces.

### Time and local-coordinate evaluation

For an in-plane voxel coordinate \(\mathbf{x}_{xy}\), the closest scan
fraction is

$$
u=
\operatorname{clip}\left[
\frac{(\mathbf{x}_{xy}-\mathbf{s}_{j,xy})
\cdot\mathbf{d}_{j,xy}}
{\lVert\mathbf{d}_{j,xy}\rVert^2},
0,1
\right].
$$

The local time is \(\tau=uT_j\), the path position is
\(\mathbf{p}_j=\mathbf{s}_j+u\mathbf{d}_j\), and

$$
y=(\mathbf{x}-\mathbf{p}_j)\cdot\mathbf{e}_{0,j},
\qquad
z=x_z-p_{j,z}.
$$

For a horizontal vector, \(u\), \(W\), \(D\), \(H\), and \(y\) are invariant
along a voxel \(z\)-column and are evaluated once per candidate vector-column
pair.

### One-voxel boundary band

The boundary label is based on radial distance from the melt-pool center. For
\(r=\sqrt{y^2+z^2}\), let

$$
s_-=\max\left(0,1-\frac{\Delta+\epsilon}{r}\right),
\qquad
s_+=1+\frac{\Delta+\epsilon}{r},
$$

where \(\epsilon=10^{-12}\) m is a numerical tolerance. A voxel is classified
as belonging to the one-voxel boundary band when

$$
F(s_-y,s_-z)\leq1
\quad\mathrm{and}\quad
F(s_+y,s_+z)\geq1.
$$

The melt predicate is evaluated at a scale infinitesimally inside the voxel
radius. Common exponents \(n=1\) and \(n=2\) use multiplication-only branches;
general positive exponents use a power evaluation. No Newton iteration or
root solve occurs in the production classification kernel.

### Conservative vertical pruning

Before traversing \(z\), Raptor contracts the transverse coordinate by one
voxel:

$$
y_c=\max(0,|y|-\Delta-\epsilon),
\qquad
R=\max\left[0,1-\left(\frac{y_c}{W/2}\right)^2\right].
$$

If \(y_c>W/2\), the candidate cannot interact with the column. Otherwise the
only potentially interacting vertical interval is conservatively bounded by

$$
z_{\min}=-D R^{1/n_D}-\Delta-\epsilon,
\qquad
z_{\max}= H R^{1/n_H}+\Delta+\epsilon.
$$

This dynamic interval is intersected with the vector AABB and RVE bounds.

## Spatial index and parallel traversal

The \(x\)-\(y\) plane is partitioned into square tiles. The default tile width
in voxels is

$$
n_{\mathrm{tile}}=
\min\left[
\max(N_x,N_y),
\max\left(16,\operatorname{round}
\frac{80\ \mu\mathrm{m}}{\Delta}\right)
\right].
$$

Only vectors whose three-dimensional AABBs overlap the RVE are retained.
Their in-plane AABBs are mapped to overlapping tiles and stored in compressed
offset/index arrays. Candidate order follows scan-vector order and is
deterministic.

Tiles are lookup bins, not processor-owned subdomains. Numba parallelizes over
individual \(x\)-\(y\) voxel columns, and each column consults its containing
tile for candidate vectors. Consequently, the default width is independent of
thread count. An explicit `tile_width` in metres overrides the automatic
80-micrometre target and bypasses its 16-voxel minimum.

Numba parallelizes over the \(N_xN_y\) columns. Each worker:

1. reconstructs \(x\) and \(y\) from integer indices;
2. obtains the ordered candidate list for the containing tile;
3. applies AABB and oriented in-plane rejection tests;
4. projects the column onto the vector and interpolates the three spectral
   dimensions;
5. computes the conservative \(z\) interval; and
6. classifies only the contiguous \(z\) values inside that interval.

The phase codes are:

| Code | Meaning |
|---:|---|
| 0 | no retained melt-pool interaction; treated as defect |
| 1 | melted interior |
| 2 | one-voxel melt-pool boundary |
| 3 | boundary interaction at a voxel previously carrying a boundary code |

Updates follow ordered vector traversal. A melt-interior update writes code 1,
a boundary update writes code 2, and a boundary update applied to a previously
boundary-coded voxel writes code 3.

## Pore morphology

Defects are phase-zero voxels. For sparse defect fields, Raptor collects only
their flat indices and labels 26-connected components with a union-find
algorithm. Single-voxel components are removed. For component \(c\) with
\(n_c\) voxels,

$$
V_c=n_c\Delta^3,
\qquad
d_{\mathrm{eq},c}=
\left(\frac{6V_c}{\pi}\right)^{1/3}.
$$

Sparse processing is selected when the number of defect voxels is no greater
than both 1% of the RVE and \(10^7\), and when all requested properties are
among area/volume, centroid, equivalent diameter, and label. Other property
sets use the dense scikit-image implementation.

## Output representation

The phase field is written as signed 8-bit VTK image data. Raptor's in-memory
array has \(z\) contiguous, so VTK dimensions and a direction matrix permute
the exposed axes without a full gigabyte-scale transpose. Files use LZ4
compression, 1 MiB blocks, and inline base64 binary data. Inline encoding is
retained for compatibility with older ParaView/VTK readers of very large
arrays.

## Computational complexity and memory

Spectral-table construction costs

$$
O\left[
\sum_j N_j(M_W+M_D+M_H)
\right].
$$

After construction, spectral lookup is constant work per accepted
vector-column pair. The voxel kernel cost is governed by the tile candidate
count, in-plane rejection rate, and length of the pruned vertical interval,
rather than by the full product of voxels and path vectors.

Primary memory is

$$
O(N) \text{ bytes}
+
O\left(3\sum_jN_j\right)\text{ Float32 values}
+
O(N_{\mathrm{candidates}})
$$

for the phase field, spectral table, and spatial index, respectively. The
eliminated explicit coordinate array would require \(24N\) bytes in Float64.

## Current constraints

- Production scan vectors must be horizontal and use global \(z\) as their
  local build direction.
- The guarded Float32 spectral bound must not exceed the requested
  `spectral_error_fraction`; the default is \(0.25\Delta\).
- The packed spectral table must not exceed the requested memory limit; the
  default is 256 MiB.
- The melt-pool history is represented by stationary cosine modes over each
  vector; nonstationary high-fidelity histories require preprocessing or a
  future direct-history interface.
- Phase-code updates are deterministic but ordered; changing vector order can
  change boundary/intersection labels.
