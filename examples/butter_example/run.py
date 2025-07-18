import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import welch
from meltpool_butter import MeltPoolFiltration


# GLOBALS
MU = 124.0          # Mean depth in micrometers
SIGMA_TOTAL = 12.0  # Target TOTAL standard deviation
# LPBF Process Parameters
SCAN_SPEED = 1.0  # m/s
# Time series parameters
FS = 250000
DURATION = 0.08
# Instantiate object
mp_filter = MeltPoolFiltration(
    MU,SIGMA_TOTAL,SCAN_SPEED,[FS,DURATION]
)
# Define physical scales
POWDER_EFFECT_PARAMS = [100e-6,None,0.2]
MELTPOOL_EFECT_PARAMS = [520e-6,None,1]
# Add effects to object
mp_filter.add_effect('Powder',POWDER_EFFECT_PARAMS)
mp_filter.add_effect('MeltPool',MELTPOOL_EFECT_PARAMS)
# Initialize physical effects by compute frequencies from length scales / sigma contributions
mp_filter.initialize()
# Generate fluctuations with the filtration process with some noise scale.
NOISE_SCALE = 1
final_series = mp_filter.generate_fluctuations(NOISE_SCALE)


# Comapre to Naive Random Sampling (e.g. Northwestern group)
# Each time step is an independent random draw from the target normal distribution.
print("\n--- Generating Naive Random Sampling Model ---")
naive_series = np.random.normal(loc=MU, scale=SIGMA_TOTAL, size=mp_filter.n_points)

# --- 5. Plot the Results ---
fig = plt.figure(figsize=(20, 12))
fig.suptitle("Comparison: Physics-Informed vs. Naive Random Sampling", fontsize=18)
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
zoom_ms = 8.0
zoom_pts = int(zoom_ms / 1000 / mp_filter.dt)

# Plot 1: Individual Components of the Physics-Informed Model
ax1 = fig.add_subplot(gs[0, 0])
for name, series in mp_filter.component_series.items():
    ax1.plot(mp_filter.t[:zoom_pts] * 1000, series[:zoom_pts] + MU, lw=1.2, alpha=0.9, label=f"{name.split(' ')[0]} (σ={np.std(series):.2f})")
ax1.set_title("Components of the Physics-Informed Signal")
ax1.set_xlabel("Time (ms)")
ax1.legend()
ax1.grid(True, alpha=0.6)

# Plot 2: Time Series
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(mp_filter.t[:zoom_pts] * 1000, final_series[:zoom_pts], lw=2, alpha=1, label='Physics-Informed Signal')
ax2.plot(mp_filter.t[:zoom_pts] * 1000, naive_series[:zoom_pts], lw=1.0, alpha=0.7, label='Naive Random Signal', zorder=-1)
ax2.set_title("Time Series")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel(f"Melt Pool Dimension ({chr(956)}m)")
ax2.legend()
ax2.grid(True, alpha=0.6)
ax2.set_ylim(0,200)

# Plot 3: Power Spectral Density (PSD)
ax3 = fig.add_subplot(gs[1, 0])
f_det, Pxx_det = welch(final_series, mp_filter.fs, nperseg=8192)
f_naive, Pxx_naive = welch(naive_series, mp_filter.fs, nperseg=8192)
ax3.semilogy(f_det / 1000, Pxx_det, lw=2, label='Physics-Informed PSD')
ax3.semilogy(f_naive / 1000, Pxx_naive, lw=2, linestyle='--', label='Naive (White Noise) PSD')
ax3.set_title("Power Spectral Density")
ax3.set_xlabel("Frequency (kHz)")
ax3.set_ylabel("Power")
ax3.legend()
ax3.grid(True, alpha=0.6)

# Plot 2: Histogram vs. Target Distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(final_series, bins=50, density=True, alpha=0.5, label='Physics-Informed Data')
ax4.hist(naive_series, bins=50, density=True, alpha=0.5, label='Naive Data')
x_range = np.linspace(MU - 4*SIGMA_TOTAL, MU + 4*SIGMA_TOTAL, 1000)
ax4.plot(x_range, norm.pdf(x_range, MU, SIGMA_TOTAL), 'r--', lw=2, label='Target Normal PDF')
ax4.set_title("Distribution Comparison")
ax4.set_xlabel(f"Melt Pool Dimension ({chr(956)}m)")
ax4.legend()
ax4.grid(True, alpha=0.6)


plt.show()