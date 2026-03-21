#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace raptor {

enum class FloatingPrecision {
  double_precision,
  single_precision
};

using Vec3 = std::array<double, 3>;
using BoundBox = std::array<Vec3, 2>;

template <typename Real>
using Vec3T = std::array<Real, 3>;

template <typename Real>
using BoundBoxT = std::array<Vec3T<Real>, 2>;

enum class PorosityKernelVariant {
  baseline,
  cached,
  bitpacked_repeats,
  seed_batch4,
  team_tile_seed_batch4,
  team_bitpacked_repeat64
};

template <typename Real>
struct ModeDataT {
  Real amplitude = static_cast<Real>(0);
  Real frequency = static_cast<Real>(0);
  Real phase = static_cast<Real>(0);
};

template <typename Real>
struct MeltPoolT {
  std::vector<ModeDataT<Real>> width_oscillations;
  std::vector<ModeDataT<Real>> depth_oscillations;
  std::vector<ModeDataT<Real>> height_oscillations;
  Real width_shape_factor = static_cast<Real>(2);
  Real height_shape_factor = static_cast<Real>(2);
  Real depth_shape_factor = static_cast<Real>(2);
  Real width_max = static_cast<Real>(0);
  Real depth_max = static_cast<Real>(0);
  Real height_max = static_cast<Real>(0);
  bool enable_random_phases = false;
  Real width_mean = static_cast<Real>(0);
  Real depth_mean = static_cast<Real>(0);
  Real height_mean = static_cast<Real>(0);
  std::size_t n_modes = 0;
};

template <typename Real>
struct PathVectorT {
  Vec3T<Real> start_point{};
  Vec3T<Real> end_point{};
  Real start_time = static_cast<Real>(0);
  Real end_time = static_cast<Real>(0);
  Real duration = static_cast<Real>(0);
  Vec3T<Real> distance{};
  std::array<Real, 6> aabb{};
  Vec3T<Real> e0{};
  Vec3T<Real> e1{};
  Vec3T<Real> e2{};
  Real L0 = static_cast<Real>(0);
  Real L1 = static_cast<Real>(0);
  Real L2 = static_cast<Real>(0);
  std::vector<Real> phases;
  Vec3T<Real> centroid{};

  PathVectorT() = default;
  PathVectorT(const Vec3T<Real>& start_point_in, const Vec3T<Real>& end_point_in,
              Real start_time_in, Real end_time_in);

  // Build the local orthonormal frame used by the melt-mask kernel.
  void setCoordinateFrame();

  // Attach melt-pool dependent phases and bounding volumes to this vector.
  void setMeltPoolProperties(const MeltPoolT<Real>& melt_pool, std::mt19937_64& rng);
};

template <typename Real>
struct GridT {
  Real resolution = static_cast<Real>(0);
  Vec3T<Real> origin{};
  std::array<std::size_t, 3> shape{};
  std::size_t n_voxels = 0;
  std::vector<Real> voxels;

  GridT() = default;
  GridT(Real voxel_resolution, const std::optional<BoundBoxT<Real>>& bound_box,
        const std::vector<PathVectorT<Real>>* path_vectors);
};

template <typename Real>
struct MeltPoolComponentInputT {
  std::vector<std::vector<Real>> data;
  std::size_t n_modes = 0;
  Real scale = static_cast<Real>(1);
  Real shape_factor = static_cast<Real>(2);
};

using ModeData = ModeDataT<double>;
using MeltPool = MeltPoolT<double>;
using PathVector = PathVectorT<double>;
using Grid = GridT<double>;
using MeltPoolComponentInput = MeltPoolComponentInputT<double>;

struct MorphologyTable {
  std::vector<std::string> headers;
  std::vector<std::vector<std::string>> rows;
};

// Convert a config or environment string into a porosity-kernel variant selector.
PorosityKernelVariant parsePorosityKernelVariant(std::string_view value);

// Return the stable config string for a porosity-kernel variant.
const char* toString(PorosityKernelVariant variant);

// Convert a config or environment string into a float-or-double execution selector.
FloatingPrecision parseFloatingPrecision(std::string_view value);

// Return the stable config string for a float-or-double execution selector.
const char* toString(FloatingPrecision precision);

}  // namespace raptor
