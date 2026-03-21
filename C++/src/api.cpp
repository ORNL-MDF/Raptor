#include "raptor/api.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace raptor {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Convert raw numeric rows into the fixed-width time-series view expected by the DFT helper.
template <typename Real>
std::vector<std::array<Real, 2>> toTimeSeries(const std::vector<std::vector<Real>>& rows) {
  std::vector<std::array<Real, 2>> result;
  result.reserve(rows.size());
  for (const std::vector<Real>& row : rows) {
    if (row.size() != 2) {
      throw std::invalid_argument("Time-series data must have exactly two columns.");
    }
    result.push_back({row[0], row[1]});
  }
  return result;
}

// Rotate a 2D point around the hatch-center used to generate successive layers.
template <typename Real>
std::array<Real, 2> rotatePoint(const std::array<Real, 2>& point,
                                const std::array<Real, 2>& center, Real angle) {
  const Real cosine = static_cast<Real>(std::cos(angle));
  const Real sine = static_cast<Real>(std::sin(angle));
  const Real local_x = point[0] - center[0];
  const Real local_y = point[1] - center[1];
  return {cosine * local_x - sine * local_y + center[0],
          sine * local_x + cosine * local_y + center[1]};
}

}  // namespace

template <typename Real>
GridT<Real> createGrid(Real voxel_resolution, const std::optional<BoundBoxT<Real>>& bound_box,
                       const std::vector<PathVectorT<Real>>* path_vectors) {
  // Preserve the Python API wrapper semantics around Grid construction.
  return GridT<Real>(voxel_resolution, bound_box, path_vectors);
}

template <typename Real>
std::vector<PathVectorT<Real>> createPathVectors(const BoundBoxT<Real>& bound_box, Real power,
                                                 Real scan_speed, Real hatch_spacing,
                                                 Real layer_height, Real rotation_degrees,
                                                 Real scan_extension, int extra_layers) {
  (void)power;

  // Recreate the Python scan-path builder without introducing extra filtering behavior.
  const Vec3T<Real> min_point = bound_box[0];
  const Vec3T<Real> max_point = bound_box[1];
  const Vec3T<Real> dimensions = {max_point[0] - min_point[0], max_point[1] - min_point[1],
                                  max_point[2] - min_point[2]};
  const std::array<Real, 2> center = {(min_point[0] + max_point[0]) / static_cast<Real>(2),
                                      (min_point[1] + max_point[1]) / static_cast<Real>(2)};
  const Real rotation =
      rotation_degrees * static_cast<Real>(kPi / 180.0);
  const int n_layers =
      static_cast<int>(std::floor(dimensions[2] / layer_height) + 1.0) + extra_layers;

  const Real xmin = min_point[0] - scan_extension;
  const Real xmax = max_point[0] + scan_extension;
  const Real ymin = min_point[1] - scan_extension;
  const Real ymax = max_point[1] + scan_extension;

  std::vector<std::array<Real, 2>> base_starts;
  std::vector<std::array<Real, 2>> base_ends;
  for (Real y = ymin; y < ymax; y += hatch_spacing) {
    base_starts.push_back({xmin, y});
    base_ends.push_back({xmax, y});
  }

  std::vector<std::vector<PathVectorT<Real>>> path_vector_layers(n_layers + 1);
  Real time_offset = static_cast<Real>(0);
  for (int layer = 0; layer <= n_layers; ++layer) {
    std::vector<PathVectorT<Real>> active_vectors;
    Real layer_time = time_offset;
    const Real angle = static_cast<Real>(layer) * rotation;

    for (std::size_t line = 0; line < base_starts.size(); ++line) {
      const std::array<Real, 2> rotated_start =
          layer == 0 ? base_starts[line] : rotatePoint(base_starts[line], center, angle);
      const std::array<Real, 2> rotated_end =
          layer == 0 ? base_ends[line] : rotatePoint(base_ends[line], center, angle);

      const Vec3T<Real> vector_start = {rotated_start[0], rotated_start[1],
                                        static_cast<Real>(layer) * layer_height};
      const Vec3T<Real> vector_end = {rotated_end[0], rotated_end[1],
                                      static_cast<Real>(layer) * layer_height};
      const Vec3T<Real> delta = {vector_end[0] - vector_start[0], vector_end[1] - vector_start[1],
                                 vector_end[2] - vector_start[2]};
      const Real vector_length =
          static_cast<Real>(std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                                      delta[2] * delta[2]));
      const Real scan_duration =
          scan_speed > static_cast<Real>(1.0e-12) ? vector_length / scan_speed
                                                  : static_cast<Real>(0);

      Real start_time = layer_time;
      if (layer >= 1 && active_vectors.empty() && !path_vector_layers[layer - 1].empty()) {
        start_time = path_vector_layers[layer - 1].back().start_time;
      }
      const Real end_time = start_time + scan_duration;

      active_vectors.emplace_back(vector_start, vector_end, start_time, end_time);
      active_vectors.back().setCoordinateFrame();
      layer_time = end_time;
    }

    path_vector_layers[layer] = active_vectors;
    time_offset =
        active_vectors.empty() ? static_cast<Real>(0) : active_vectors.back().end_time;
  }

  std::vector<PathVectorT<Real>> all_vectors;
  for (const std::vector<PathVectorT<Real>>& layer_vectors : path_vector_layers) {
    all_vectors.insert(all_vectors.end(), layer_vectors.begin(), layer_vectors.end());
  }
  return all_vectors;
}

template <typename Real>
std::vector<ModeDataT<Real>> computeSpectralComponents(
    const std::vector<std::array<Real, 2>>& melt_pool_data, std::size_t n_modes) {
  // Match the Python FFT contract with a direct DFT over the requested leading modes.
  if (melt_pool_data.size() < 2) {
    throw std::invalid_argument("At least two melt-pool samples are required.");
  }
  if (n_modes == 0) {
    throw std::invalid_argument("At least one spectral mode is required.");
  }

  const Real dt = melt_pool_data[1][0] - melt_pool_data[0][0];
  const std::size_t n_fft = melt_pool_data.size();
  if (n_modes > n_fft) {
    throw std::invalid_argument("Requested spectral modes exceed the available sample count.");
  }
  std::vector<ModeDataT<Real>> spectral_array(n_modes);

  Real mode0 = static_cast<Real>(0);
  for (const auto& sample : melt_pool_data) {
    mode0 += sample[1];
  }
  mode0 /= static_cast<Real>(n_fft);
  spectral_array[0] = {mode0, static_cast<Real>(0), static_cast<Real>(0)};

  for (std::size_t mode = 1; mode < n_modes; ++mode) {
    std::complex<double> coefficient(0.0, 0.0);
    for (std::size_t index = 0; index < n_fft; ++index) {
      const double angle = -2.0 * kPi * static_cast<double>(mode) *
                           static_cast<double>(index) / static_cast<double>(n_fft);
      coefficient += static_cast<double>(melt_pool_data[index][1]) *
                     std::complex<double>(std::cos(angle), std::sin(angle));
    }

    spectral_array[mode].amplitude =
        static_cast<Real>(std::abs(coefficient) / static_cast<double>(n_fft));
    spectral_array[mode].frequency =
        static_cast<Real>((1.0 / (static_cast<double>(dt) * static_cast<double>(n_fft))) *
                          static_cast<double>(mode));
    spectral_array[mode].phase = static_cast<Real>(std::arg(coefficient));
  }

  return spectral_array;
}

template <typename Real>
MeltPoolT<Real> createMeltPool(
    const std::map<std::string, MeltPoolComponentInputT<Real>>& melt_pool_dict,
    bool enable_random_phases) {
  // Normalize the three component inputs into a packed melt-pool description.
  std::size_t max_modes = 0;
  for (const auto& [_, component] : melt_pool_dict) {
    max_modes = std::max(max_modes, component.n_modes);
  }

  std::map<std::string, std::vector<ModeDataT<Real>>> processed_components;
  for (const auto& [name, component] : melt_pool_dict) {
    std::vector<ModeDataT<Real>> spectral_array;
    if (!component.data.empty() && component.data[0].size() == 2) {
      spectral_array = computeSpectralComponents<Real>(toTimeSeries(component.data), component.n_modes);
      for (ModeDataT<Real>& mode : spectral_array) {
        mode.amplitude *= component.scale;
      }
    } else if (!component.data.empty() && component.data[0].size() == 3) {
      spectral_array.reserve(component.data.size());
      for (const std::vector<Real>& row : component.data) {
        spectral_array.push_back({row[0], row[1], row[2]});
      }
    } else {
      throw std::invalid_argument("Unsupported melt-pool data shape.");
    }

    while (spectral_array.size() < max_modes) {
      spectral_array.push_back(
          {static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(0)});
    }
    processed_components[name] = spectral_array;
  }

  MeltPoolT<Real> melt_pool;
  melt_pool.width_oscillations = processed_components.at("width");
  melt_pool.depth_oscillations = processed_components.at("depth");
  melt_pool.height_oscillations = processed_components.at("height");
  melt_pool.width_shape_factor = melt_pool_dict.at("width").shape_factor;
  melt_pool.depth_shape_factor = melt_pool_dict.at("depth").shape_factor;
  melt_pool.height_shape_factor = melt_pool_dict.at("height").shape_factor;
  melt_pool.enable_random_phases = enable_random_phases;
  melt_pool.n_modes = max_modes;

  for (std::size_t mode = 0; mode < max_modes; ++mode) {
    melt_pool.width_max += melt_pool.width_oscillations[mode].amplitude;
    melt_pool.depth_max += melt_pool.depth_oscillations[mode].amplitude;
    melt_pool.height_max += melt_pool.height_oscillations[mode].amplitude;
  }
  melt_pool.width_mean =
      max_modes > 0 ? melt_pool.width_oscillations[0].amplitude : static_cast<Real>(0);
  melt_pool.depth_mean =
      max_modes > 0 ? melt_pool.depth_oscillations[0].amplitude : static_cast<Real>(0);
  melt_pool.height_mean =
      max_modes > 0 ? melt_pool.height_oscillations[0].amplitude : static_cast<Real>(0);

  return melt_pool;
}

template <typename Real>
std::vector<std::uint8_t> computePorosity(const GridT<Real>& grid,
                                          std::vector<PathVectorT<Real>>& path_vectors,
                                          const MeltPoolT<Real>& melt_pool) {
  Kokkos::Profiling::ScopedRegion region("raptor::compute_porosity");

  // Dispatch the single-run API through the reusable multi-run device workflow.
  std::random_device random_device;
  const std::uint64_t base_seed =
      (static_cast<std::uint64_t>(random_device()) << 32) ^
      static_cast<std::uint64_t>(random_device());
  PorosityRunSummary summary;
  {
    Kokkos::Profiling::ScopedRegion run_region("raptor::compute_porosity_run");
    summary = computePorosityRuns<Real>(grid, path_vectors, melt_pool, 1, base_seed, true, {},
                                        PorosityKernelVariant::baseline);
  }
  return summary.final_porosity;
}

template GridT<float> createGrid<float>(float, const std::optional<BoundBoxT<float>>&,
                                        const std::vector<PathVectorT<float>>*);
template GridT<double> createGrid<double>(double, const std::optional<BoundBoxT<double>>&,
                                          const std::vector<PathVectorT<double>>*);
template std::vector<PathVectorT<float>> createPathVectors<float>(
    const BoundBoxT<float>&, float, float, float, float, float, float, int);
template std::vector<PathVectorT<double>> createPathVectors<double>(
    const BoundBoxT<double>&, double, double, double, double, double, double, int);
template std::vector<ModeDataT<float>> computeSpectralComponents<float>(
    const std::vector<std::array<float, 2>>&, std::size_t);
template std::vector<ModeDataT<double>> computeSpectralComponents<double>(
    const std::vector<std::array<double, 2>>&, std::size_t);
template MeltPoolT<float> createMeltPool<float>(
    const std::map<std::string, MeltPoolComponentInputT<float>>&, bool);
template MeltPoolT<double> createMeltPool<double>(
    const std::map<std::string, MeltPoolComponentInputT<double>>&, bool);
template std::vector<std::uint8_t> computePorosity<float>(const GridT<float>&,
                                                          std::vector<PathVectorT<float>>&,
                                                          const MeltPoolT<float>&);
template std::vector<std::uint8_t> computePorosity<double>(const GridT<double>&,
                                                           std::vector<PathVectorT<double>>&,
                                                           const MeltPoolT<double>&);

}  // namespace raptor
