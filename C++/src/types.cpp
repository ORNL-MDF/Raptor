#include "raptor/types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace raptor {
namespace {

constexpr double kTwoPi = 6.28318530717958647692;

// Compute the axis length using the same inclusive upper-bound rule as NumPy arange.
template <typename Real>
std::size_t computeAxisLength(Real lower, Real upper, Real step) {
  return static_cast<std::size_t>(std::llround((upper - lower) / step)) + 1;
}

}  // namespace

PorosityKernelVariant parsePorosityKernelVariant(std::string_view value) {
  // Parse the shared kernel-selector strings used by the CLI config and RVE benchmark app.
  const std::string lowered(value);
  if (lowered == "baseline") {
    return PorosityKernelVariant::baseline;
  }
  if (lowered == "cached") {
    return PorosityKernelVariant::cached;
  }
  if (lowered == "seed_batch4") {
    return PorosityKernelVariant::seed_batch4;
  }
  if (lowered == "team_tile_seed_batch4") {
    return PorosityKernelVariant::team_tile_seed_batch4;
  }
  if (lowered == "auto") {
    return PorosityKernelVariant::auto_select;
  }
  throw std::invalid_argument("Unknown porosity kernel variant: " + std::string(value));
}

const char* toString(PorosityKernelVariant variant) {
  // Preserve one canonical string spelling for config parsing and benchmark reporting.
  switch (variant) {
    case PorosityKernelVariant::baseline:
      return "baseline";
    case PorosityKernelVariant::cached:
      return "cached";
    case PorosityKernelVariant::seed_batch4:
      return "seed_batch4";
    case PorosityKernelVariant::team_tile_seed_batch4:
      return "team_tile_seed_batch4";
    case PorosityKernelVariant::auto_select:
      return "auto";
  }
  throw std::invalid_argument("Invalid porosity kernel variant enum value.");
}

FloatingPrecision parseFloatingPrecision(std::string_view value) {
  // Parse the app-level floating-point selection used to instantiate the workflow.
  const std::string lowered(value);
  if (lowered == "double") {
    return FloatingPrecision::double_precision;
  }
  if (lowered == "float" || lowered == "single") {
    return FloatingPrecision::single_precision;
  }
  throw std::invalid_argument("Unknown floating precision: " + std::string(value));
}

const char* toString(FloatingPrecision precision) {
  // Preserve one canonical string spelling for config and environment reporting.
  switch (precision) {
    case FloatingPrecision::double_precision:
      return "double";
    case FloatingPrecision::single_precision:
      return "float";
  }
  throw std::invalid_argument("Invalid floating precision enum value.");
}

template <typename Real>
PathVectorT<Real>::PathVectorT(const Vec3T<Real>& start_point_in,
                               const Vec3T<Real>& end_point_in, Real start_time_in,
                               Real end_time_in)
    : start_point(start_point_in), end_point(end_point_in), start_time(start_time_in),
      end_time(end_time_in) {}

template <typename Real>
void PathVectorT<Real>::setCoordinateFrame() {
  // Derive the line direction, centroid, and duration for later culling and timing.
  distance = {end_point[0] - start_point[0], end_point[1] - start_point[1],
              end_point[2] - start_point[2]};
  centroid = {(end_point[0] + start_point[0]) / static_cast<Real>(2),
              (end_point[1] + start_point[1]) / static_cast<Real>(2),
              (end_point[2] + start_point[2]) / static_cast<Real>(2)};
  duration = end_time - start_time;

  // Build the local scan-aligned frame, matching the Python XY fallback behavior.
  const Real dx = distance[0];
  const Real dy = distance[1];
  const Real length_xy = static_cast<Real>(std::hypot(dx, dy));
  if (length_xy < static_cast<Real>(1.0e-12)) {
    e0 = {static_cast<Real>(1), static_cast<Real>(0), static_cast<Real>(0)};
    e1 = {static_cast<Real>(0), static_cast<Real>(1), static_cast<Real>(0)};
  } else {
    e0 = {-dy / length_xy, dx / length_xy, static_cast<Real>(0)};
    e1 = {dx / length_xy, dy / length_xy, static_cast<Real>(0)};
  }
  e2 = {static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(1)};
}

template <typename Real>
void PathVectorT<Real>::setMeltPoolProperties(const MeltPoolT<Real>& melt_pool,
                                              std::mt19937_64& rng) {
  // Assign per-vector phases, preserving the shared phase convention across dimensions.
  phases.assign(melt_pool.n_modes, static_cast<Real>(0));
  if (melt_pool.enable_random_phases && melt_pool.n_modes > 1) {
    std::uniform_real_distribution<double> distribution(0.0, kTwoPi);
    for (std::size_t mode = 1; mode < melt_pool.n_modes; ++mode) {
      phases[mode] = static_cast<Real>(distribution(rng));
    }
  } else {
    for (std::size_t mode = 0; mode < melt_pool.n_modes; ++mode) {
      phases[mode] = melt_pool.width_oscillations[mode].phase;
    }
  }

  // Expand the axis-aligned and oriented bounds used to prune work in the kernel.
  const Vec3T<Real> point_min = {std::min(start_point[0], end_point[0]),
                                 std::min(start_point[1], end_point[1]),
                                 std::min(start_point[2], end_point[2])};
  const Vec3T<Real> point_max = {std::max(start_point[0], end_point[0]),
                                 std::max(start_point[1], end_point[1]),
                                 std::max(start_point[2], end_point[2])};
  const Real pad_xy = melt_pool.width_max / static_cast<Real>(2);
  aabb = {point_min[0] - pad_xy, point_max[0] + pad_xy, point_min[1] - pad_xy,
          point_max[1] + pad_xy, point_min[2] - melt_pool.depth_max,
          point_max[2] + melt_pool.height_max};

  L0 = melt_pool.width_max / static_cast<Real>(2);
  L1 = static_cast<Real>(
      std::hypot(point_max[0] - point_min[0], point_max[1] - point_min[1]) /
      static_cast<Real>(2));
  L2 = std::max(melt_pool.height_max, melt_pool.depth_max);
}

template <typename Real>
GridT<Real>::GridT(Real voxel_resolution, const std::optional<BoundBoxT<Real>>& bound_box,
                   const std::vector<PathVectorT<Real>>* path_vectors) {
  // Validate the domain definition before building the point grid.
  if (voxel_resolution <= static_cast<Real>(0)) {
    throw std::invalid_argument("Voxel resolution must be a positive non-zero value.");
  }
  resolution = voxel_resolution;

  Vec3T<Real> min_corner{};
  Vec3T<Real> max_corner{};
  if (bound_box.has_value()) {
    if ((*bound_box)[1][0] <= (*bound_box)[0][0] || (*bound_box)[1][1] <= (*bound_box)[0][1] ||
        (*bound_box)[1][2] <= (*bound_box)[0][2]) {
      throw std::invalid_argument(
          "Invalid bounding box: maximum corner must be greater than minimum corner.");
    }
    min_corner = (*bound_box)[0];
    max_corner = (*bound_box)[1];
  } else if (path_vectors != nullptr && !path_vectors->empty()) {
    min_corner = (*path_vectors)[0].start_point;
    max_corner = (*path_vectors)[0].start_point;
    for (const PathVectorT<Real>& path_vector : *path_vectors) {
      for (int dim = 0; dim < 3; ++dim) {
        min_corner[dim] = std::min({min_corner[dim], path_vector.start_point[dim],
                                    path_vector.end_point[dim]});
        max_corner[dim] = std::max({max_corner[dim], path_vector.start_point[dim],
                                    path_vector.end_point[dim]});
      }
    }
  } else {
    throw std::invalid_argument(
        "Grid construction failed: provide either a bounding box or path vectors.");
  }

  origin = min_corner;

  // Store the grid metadata and let the device kernel derive voxel coordinates lazily.
  shape = {computeAxisLength(min_corner[0], max_corner[0], resolution),
           computeAxisLength(min_corner[1], max_corner[1], resolution),
           computeAxisLength(min_corner[2], max_corner[2], resolution)};
  n_voxels = shape[0] * shape[1] * shape[2];
  voxels.clear();
}

template struct PathVectorT<float>;
template struct PathVectorT<double>;
template struct GridT<float>;
template struct GridT<double>;

}  // namespace raptor
