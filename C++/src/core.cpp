#include "raptor/api.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace raptor {
namespace {

constexpr int kMaxSeedBatchLanes = 4;
constexpr int kPackedMaskWordBits = 64;
constexpr int kTeamRepeat64Lanes = 64;
constexpr int kTeamTileVoxels = 32;
constexpr int kTeamVectorChunk = 16;

constexpr int kVectorColumns = 33;
constexpr int kVectorStartOffset = 0;
constexpr int kVectorDistanceOffset = 3;
constexpr int kVectorE0Offset = 6;
constexpr int kVectorE1Offset = 9;
constexpr int kVectorE2Offset = 12;
constexpr int kVectorAabbOffset = 15;
constexpr int kVectorCentroidOffset = 21;
constexpr int kVectorL0Offset = 24;
constexpr int kVectorL1Offset = 25;
constexpr int kVectorStartTimeOffset = 26;
constexpr int kVectorEndTimeOffset = 27;
constexpr int kVectorDistSquaredOffset = 28;
constexpr int kVectorInvDistSquaredOffset = 29;
constexpr int kVectorL0SquaredOffset = 30;
constexpr int kVectorL1SquaredOffset = 31;
constexpr int kVectorDurationOffset = 32;

constexpr int kModeColumns = 9;
constexpr int kModeWidthAmplitudeOffset = 0;
constexpr int kModeWidthFrequencyOffset = 1;
constexpr int kModeDepthAmplitudeOffset = 2;
constexpr int kModeDepthFrequencyOffset = 3;
constexpr int kModeHeightAmplitudeOffset = 4;
constexpr int kModeHeightFrequencyOffset = 5;
constexpr int kModeWidthTwoPiFrequencyOffset = 6;
constexpr int kModeDepthTwoPiFrequencyOffset = 7;
constexpr int kModeHeightTwoPiFrequencyOffset = 8;

template <typename Real>
KOKKOS_INLINE_FUNCTION bool isInside(Real y, Real z, Real width, Real height, Real depth,
                                     Real height_shape_factor, Real depth_shape_factor);

// Clamp interpolation factors to the closed segment interval used by the Python solver.
template <typename Real>
KOKKOS_INLINE_FUNCTION Real clamp01(Real value) {
  return value < static_cast<Real>(0)
             ? static_cast<Real>(0)
             : (value > static_cast<Real>(1) ? static_cast<Real>(1) : value);
}

// Check whether every lane in a batched-seed kernel has already melted.
KOKKOS_INLINE_FUNCTION bool allLanesMelted(const std::uint8_t lane_melted[],
                                           int lane_count) {
  for (int lane = 0; lane < lane_count; ++lane) {
    if (lane_melted[lane] == 0) {
      return false;
    }
  }
  return true;
}

// Build the all-ones mask corresponding to the active repeat bits in one packed batch.
template <typename MaskType>
KOKKOS_INLINE_FUNCTION MaskType fullMaskForBits(int bit_count) {
  constexpr int kMaskBits = static_cast<int>(sizeof(MaskType) * 8);
  return bit_count >= kMaskBits ? static_cast<MaskType>(~static_cast<MaskType>(0))
                                : static_cast<MaskType>((static_cast<MaskType>(1) << bit_count) -
                                                        static_cast<MaskType>(1));
}

// Convert a repeat count into the number of packed 64-bit mask words needed per voxel.
KOKKOS_INLINE_FUNCTION int packedMaskWordCount(int repeat_count) {
  return (repeat_count + kPackedMaskWordBits - 1) / kPackedMaskWordBits;
}

// Fill the packed path-vector row with geometry and melt-pool bounds that are stable across runs.
template <typename Real>
void packStaticVectorRow(Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace>& vector_data_host,
                         int vector_index, const PathVectorT<Real>& path_vector,
                         const MeltPoolT<Real>& melt_pool) {
  const Vec3T<Real> point_min = {std::min(path_vector.start_point[0], path_vector.end_point[0]),
                                 std::min(path_vector.start_point[1], path_vector.end_point[1]),
                                 std::min(path_vector.start_point[2], path_vector.end_point[2])};
  const Vec3T<Real> point_max = {std::max(path_vector.start_point[0], path_vector.end_point[0]),
                                 std::max(path_vector.start_point[1], path_vector.end_point[1]),
                                 std::max(path_vector.start_point[2], path_vector.end_point[2])};
  const Real pad_xy = melt_pool.width_max / static_cast<Real>(2);
  const Real l0 = melt_pool.width_max / static_cast<Real>(2);
  const Real l1 = static_cast<Real>(
      std::hypot(point_max[0] - point_min[0], point_max[1] - point_min[1]) /
      static_cast<Real>(2));
  const Real dist_x = path_vector.distance[0];
  const Real dist_y = path_vector.distance[1];
  const Real dist_z = path_vector.distance[2];
  const Real dist_sqr = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;

  vector_data_host(vector_index, kVectorStartOffset + 0) = path_vector.start_point[0];
  vector_data_host(vector_index, kVectorStartOffset + 1) = path_vector.start_point[1];
  vector_data_host(vector_index, kVectorStartOffset + 2) = path_vector.start_point[2];
  vector_data_host(vector_index, kVectorDistanceOffset + 0) = dist_x;
  vector_data_host(vector_index, kVectorDistanceOffset + 1) = dist_y;
  vector_data_host(vector_index, kVectorDistanceOffset + 2) = dist_z;
  vector_data_host(vector_index, kVectorE0Offset + 0) = path_vector.e0[0];
  vector_data_host(vector_index, kVectorE0Offset + 1) = path_vector.e0[1];
  vector_data_host(vector_index, kVectorE0Offset + 2) = path_vector.e0[2];
  vector_data_host(vector_index, kVectorE1Offset + 0) = path_vector.e1[0];
  vector_data_host(vector_index, kVectorE1Offset + 1) = path_vector.e1[1];
  vector_data_host(vector_index, kVectorE1Offset + 2) = path_vector.e1[2];
  vector_data_host(vector_index, kVectorE2Offset + 0) = path_vector.e2[0];
  vector_data_host(vector_index, kVectorE2Offset + 1) = path_vector.e2[1];
  vector_data_host(vector_index, kVectorE2Offset + 2) = path_vector.e2[2];
  vector_data_host(vector_index, kVectorAabbOffset + 0) = point_min[0] - pad_xy;
  vector_data_host(vector_index, kVectorAabbOffset + 1) = point_max[0] + pad_xy;
  vector_data_host(vector_index, kVectorAabbOffset + 2) = point_min[1] - pad_xy;
  vector_data_host(vector_index, kVectorAabbOffset + 3) = point_max[1] + pad_xy;
  vector_data_host(vector_index, kVectorAabbOffset + 4) = point_min[2] - melt_pool.depth_max;
  vector_data_host(vector_index, kVectorAabbOffset + 5) = point_max[2] + melt_pool.height_max;
  vector_data_host(vector_index, kVectorCentroidOffset + 0) = path_vector.centroid[0];
  vector_data_host(vector_index, kVectorCentroidOffset + 1) = path_vector.centroid[1];
  vector_data_host(vector_index, kVectorCentroidOffset + 2) = path_vector.centroid[2];
  vector_data_host(vector_index, kVectorL0Offset) = l0;
  vector_data_host(vector_index, kVectorL1Offset) = l1;
  vector_data_host(vector_index, kVectorStartTimeOffset) = path_vector.start_time;
  vector_data_host(vector_index, kVectorEndTimeOffset) = path_vector.end_time;
  vector_data_host(vector_index, kVectorDistSquaredOffset) = dist_sqr;
  vector_data_host(vector_index, kVectorInvDistSquaredOffset) =
      dist_sqr > static_cast<Real>(1.0e-24) ? static_cast<Real>(1) / dist_sqr
                                            : static_cast<Real>(0);
  vector_data_host(vector_index, kVectorL0SquaredOffset) = l0 * l0;
  vector_data_host(vector_index, kVectorL1SquaredOffset) = l1 * l1;
  vector_data_host(vector_index, kVectorDurationOffset) =
      path_vector.end_time - path_vector.start_time;
}

// Pack the melt-pool mode amplitudes and frequencies into one dense matrix for the device.
template <typename Real>
void packModeData(Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace>& mode_data_host,
                  const MeltPoolT<Real>& melt_pool, int n_modes) {
  for (int mode = 0; mode < n_modes; ++mode) {
    const Real width_frequency = melt_pool.width_oscillations[mode].frequency;
    const Real depth_frequency = melt_pool.depth_oscillations[mode].frequency;
    const Real height_frequency = melt_pool.height_oscillations[mode].frequency;
    mode_data_host(mode, kModeWidthAmplitudeOffset) = melt_pool.width_oscillations[mode].amplitude;
    mode_data_host(mode, kModeWidthFrequencyOffset) = width_frequency;
    mode_data_host(mode, kModeDepthAmplitudeOffset) = melt_pool.depth_oscillations[mode].amplitude;
    mode_data_host(mode, kModeDepthFrequencyOffset) = depth_frequency;
    mode_data_host(mode, kModeHeightAmplitudeOffset) =
        melt_pool.height_oscillations[mode].amplitude;
    mode_data_host(mode, kModeHeightFrequencyOffset) = height_frequency;
    mode_data_host(mode, kModeWidthTwoPiFrequencyOffset) =
        static_cast<Real>(2.0 * 3.14159265358979323846) * width_frequency;
    mode_data_host(mode, kModeDepthTwoPiFrequencyOffset) =
        static_cast<Real>(2.0 * 3.14159265358979323846) * depth_frequency;
    mode_data_host(mode, kModeHeightTwoPiFrequencyOffset) =
        static_cast<Real>(2.0 * 3.14159265358979323846) * height_frequency;
  }
}

// Populate prepared phases from the host-side PathVector list for the direct melt-mask API.
template <typename Real>
void packPreparedPhases(Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace>& phases_host,
                        const std::vector<PathVectorT<Real>>& path_vectors,
                        const MeltPoolT<Real>& melt_pool, int n_vectors, int n_modes) {
  for (int vector = 0; vector < n_vectors; ++vector) {
    for (int mode = 0; mode < n_modes; ++mode) {
      const bool has_phase = static_cast<int>(path_vectors[vector].phases.size()) > mode;
      phases_host(vector, mode) =
          has_phase ? path_vectors[vector].phases[mode] : melt_pool.width_oscillations[mode].phase;
    }
  }
}

// Hash the run seed, vector id, and mode id into a reproducible pseudo-random phase.
template <typename Real>
KOKKOS_INLINE_FUNCTION Real phaseFromSeed(std::uint64_t seed, int vector, int mode) {
  std::uint64_t value = seed ^ (static_cast<std::uint64_t>(vector) << 32) ^
                        static_cast<std::uint64_t>(mode);
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  value ^= value >> 31;
  const double unit = static_cast<double>(value >> 11) * (1.0 / 9007199254740992.0);
  return static_cast<Real>(2.0 * 3.14159265358979323846 * unit);
}

// Route float CUDA builds through NVIDIA fast trig intrinsics while preserving the generic path
// for double precision and non-CUDA backends.
template <typename Real>
KOKKOS_INLINE_FUNCTION Real fastCos(Real angle) {
  return Kokkos::cos(angle);
}

KOKKOS_INLINE_FUNCTION float fastCos(float angle) {
#if defined(__CUDA_ARCH__)
  return __cosf(angle);
#else
  return Kokkos::cos(angle);
#endif
}

template <typename Real>
KOKKOS_INLINE_FUNCTION void fastSinCos(Real angle, Real& sine, Real& cosine) {
  sine = Kokkos::sin(angle);
  cosine = Kokkos::cos(angle);
}

KOKKOS_INLINE_FUNCTION void fastSinCos(float angle, float& sine, float& cosine) {
#if defined(__CUDA_ARCH__)
  __sincosf(angle, &sine, &cosine);
#else
  sine = Kokkos::sin(angle);
  cosine = Kokkos::cos(angle);
#endif
}

// Fill the phase matrix on device so repeat runs only change the seed and reuse static buffers.
template <typename Real, typename PhaseView>
void fillRandomPhases(const PhaseView& phases_view, int n_vectors, int n_modes,
                      std::uint64_t seed) {
  Kokkos::parallel_for(
      "raptor::prepare_phases", Kokkos::RangePolicy<>(0, n_vectors * n_modes),
      KOKKOS_LAMBDA(const int index) {
        const int vector = index / n_modes;
        const int mode = index % n_modes;
        phases_view(vector, mode) =
            mode == 0 ? static_cast<Real>(0) : phaseFromSeed<Real>(seed, vector, mode);
      });
}

// Evaluate the per-mode oscillations for one vector while sharing geometry across seed lanes.
template <typename Real, typename ModeView, typename PhaseAccessor>
KOKKOS_INLINE_FUNCTION bool accumulateModesForLanes(
    const ModeView& mode_data_view, int n_modes, int vector_index, Real time, Real local_y,
    Real local_z, Real height_shape_factor, Real depth_shape_factor,
    const PhaseAccessor& phase_accessor, int lane_count, std::uint8_t lane_melted[]) {
  Real width[kMaxSeedBatchLanes] = {static_cast<Real>(0), static_cast<Real>(0),
                                    static_cast<Real>(0), static_cast<Real>(0)};
  Real depth[kMaxSeedBatchLanes] = {static_cast<Real>(0), static_cast<Real>(0),
                                    static_cast<Real>(0), static_cast<Real>(0)};
  Real height[kMaxSeedBatchLanes] = {static_cast<Real>(0), static_cast<Real>(0),
                                     static_cast<Real>(0), static_cast<Real>(0)};

  for (int mode = 0; mode < n_modes; ++mode) {
    const Real width_amplitude = mode_data_view(mode, kModeWidthAmplitudeOffset);
    const Real depth_amplitude = mode_data_view(mode, kModeDepthAmplitudeOffset);
    const Real height_amplitude = mode_data_view(mode, kModeHeightAmplitudeOffset);
    const Real width_frequency = mode_data_view(mode, kModeWidthTwoPiFrequencyOffset);
    const Real depth_frequency = mode_data_view(mode, kModeDepthTwoPiFrequencyOffset);
    const Real height_frequency = mode_data_view(mode, kModeHeightTwoPiFrequencyOffset);

    for (int lane = 0; lane < lane_count; ++lane) {
      if (lane_melted[lane] != 0) {
        continue;
      }
      const Real phase = phase_accessor(vector_index, mode, lane);
      width[lane] += width_amplitude * fastCos(time * width_frequency + phase);
      depth[lane] += depth_amplitude * fastCos(time * depth_frequency + phase);
      height[lane] += height_amplitude * fastCos(time * height_frequency + phase);
    }
  }

  for (int lane = 0; lane < lane_count; ++lane) {
    if (lane_melted[lane] == 0 &&
        isInside<Real>(local_y, local_z, width[lane], height[lane], depth[lane],
                       height_shape_factor, depth_shape_factor)) {
      lane_melted[lane] = 1;
    }
  }
  return allLanesMelted(lane_melted, lane_count);
}

// Evaluate one packed path vector against the current voxel for one or more seed lanes.
template <typename Real, typename RowAccessor, typename ModeView, typename PhaseAccessor>
KOKKOS_INLINE_FUNCTION bool evaluateVoxelAgainstPackedVector(
    const RowAccessor& vector_value, int vector_index, Real vx, Real vy, Real vz,
    const ModeView& mode_data_view, int n_modes, Real height_shape_factor,
    Real depth_shape_factor, const PhaseAccessor& phase_accessor, int lane_count,
    std::uint8_t lane_melted[]) {
  if (vx < vector_value(kVectorAabbOffset + 0) || vx > vector_value(kVectorAabbOffset + 1) ||
      vy < vector_value(kVectorAabbOffset + 2) || vy > vector_value(kVectorAabbOffset + 3) ||
      vz < vector_value(kVectorAabbOffset + 4) || vz > vector_value(kVectorAabbOffset + 5)) {
    return false;
  }

  const Real vec_cx = vx - vector_value(kVectorCentroidOffset + 0);
  const Real vec_cy = vy - vector_value(kVectorCentroidOffset + 1);
  const Real vec_cz = vz - vector_value(kVectorCentroidOffset + 2);

  const Real dot_e0 = vec_cx * vector_value(kVectorE0Offset + 0) +
                      vec_cy * vector_value(kVectorE0Offset + 1) +
                      vec_cz * vector_value(kVectorE0Offset + 2);
  if ((dot_e0 * dot_e0) > vector_value(kVectorL0SquaredOffset)) {
    return false;
  }

  const Real dot_e1 = vec_cx * vector_value(kVectorE1Offset + 0) +
                      vec_cy * vector_value(kVectorE1Offset + 1) +
                      vec_cz * vector_value(kVectorE1Offset + 2);
  if ((dot_e1 * dot_e1) > vector_value(kVectorL1SquaredOffset)) {
    return false;
  }

  Real time_fraction = static_cast<Real>(0);
  const Real inv_dist_sqr = vector_value(kVectorInvDistSquaredOffset);
  if (inv_dist_sqr > static_cast<Real>(0)) {
    const Real vec_sx = vx - vector_value(kVectorStartOffset + 0);
    const Real vec_sy = vy - vector_value(kVectorStartOffset + 1);
    const Real vec_sz = vz - vector_value(kVectorStartOffset + 2);
    const Real dot_dist = vec_sx * vector_value(kVectorDistanceOffset + 0) +
                          vec_sy * vector_value(kVectorDistanceOffset + 1) +
                          vec_sz * vector_value(kVectorDistanceOffset + 2);
    time_fraction = dot_dist * inv_dist_sqr;
  }

  time_fraction = clamp01(time_fraction);
  const Real time =
      vector_value(kVectorStartTimeOffset) + time_fraction * vector_value(kVectorDurationOffset);
  const Real vec_path_x =
      vx - (vector_value(kVectorStartOffset + 0) +
            time_fraction * vector_value(kVectorDistanceOffset + 0));
  const Real vec_path_y =
      vy - (vector_value(kVectorStartOffset + 1) +
            time_fraction * vector_value(kVectorDistanceOffset + 1));
  const Real vec_path_z =
      vz - (vector_value(kVectorStartOffset + 2) +
            time_fraction * vector_value(kVectorDistanceOffset + 2));

  const Real local_y = vec_path_x * vector_value(kVectorE0Offset + 0) +
                       vec_path_y * vector_value(kVectorE0Offset + 1) +
                       vec_path_z * vector_value(kVectorE0Offset + 2);
  const Real local_z = vec_path_x * vector_value(kVectorE2Offset + 0) +
                       vec_path_y * vector_value(kVectorE2Offset + 1) +
                       vec_path_z * vector_value(kVectorE2Offset + 2);

  return accumulateModesForLanes<Real>(mode_data_view, n_modes, vector_index, time, local_y,
                                       local_z, height_shape_factor, depth_shape_factor,
                                       phase_accessor, lane_count, lane_melted);
}

// Launch the original voxel-major melt-mask kernel as the correctness baseline.
template <typename Real, typename VectorView, typename PhaseView, typename ModeView,
          typename MeltedView>
void launchMeltMaskKernelBaseline(const GridT<Real>& grid, const VectorView& vector_data_view,
                                  const PhaseView& phases_view, const ModeView& mode_data_view,
                                  int n_voxels, int n_vectors, int n_modes,
                                  Real height_shape_factor, Real depth_shape_factor,
                                  const MeltedView& melted_view) {
  const int ny = static_cast<int>(grid.shape[1]);
  const int nz = static_cast<int>(grid.shape[2]);
  const Real origin_x = grid.origin[0];
  const Real origin_y = grid.origin[1];
  const Real origin_z = grid.origin[2];
  const Real resolution = grid.resolution;

  Kokkos::parallel_for(
      "raptor::compute_melt_mask", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index) {
        const int z_index = voxel_index % nz;
        const int yz_index = voxel_index / nz;
        const int y_index = yz_index % ny;
        const int x_index = yz_index / ny;
        const Real vx = origin_x + static_cast<Real>(x_index) * resolution;
        const Real vy = origin_y + static_cast<Real>(y_index) * resolution;
        const Real vz = origin_z + static_cast<Real>(z_index) * resolution;
        std::uint8_t is_voxel_melted = 0;

        for (int vector = 0; vector < n_vectors; ++vector) {
          if (vx < vector_data_view(vector, kVectorAabbOffset + 0) ||
              vx > vector_data_view(vector, kVectorAabbOffset + 1) ||
              vy < vector_data_view(vector, kVectorAabbOffset + 2) ||
              vy > vector_data_view(vector, kVectorAabbOffset + 3) ||
              vz < vector_data_view(vector, kVectorAabbOffset + 4) ||
              vz > vector_data_view(vector, kVectorAabbOffset + 5)) {
            continue;
          }

          const Real vec_cx = vx - vector_data_view(vector, kVectorCentroidOffset + 0);
          const Real vec_cy = vy - vector_data_view(vector, kVectorCentroidOffset + 1);
          const Real vec_cz = vz - vector_data_view(vector, kVectorCentroidOffset + 2);

          const Real dot_e0 = vec_cx * vector_data_view(vector, kVectorE0Offset + 0) +
                              vec_cy * vector_data_view(vector, kVectorE0Offset + 1) +
                              vec_cz * vector_data_view(vector, kVectorE0Offset + 2);
          if ((dot_e0 * dot_e0) >
              (vector_data_view(vector, kVectorL0Offset) *
               vector_data_view(vector, kVectorL0Offset))) {
            continue;
          }

          const Real dot_e1 = vec_cx * vector_data_view(vector, kVectorE1Offset + 0) +
                              vec_cy * vector_data_view(vector, kVectorE1Offset + 1) +
                              vec_cz * vector_data_view(vector, kVectorE1Offset + 2);
          if ((dot_e1 * dot_e1) >
              (vector_data_view(vector, kVectorL1Offset) *
               vector_data_view(vector, kVectorL1Offset))) {
            continue;
          }

          const Real dist_x = vector_data_view(vector, kVectorDistanceOffset + 0);
          const Real dist_y = vector_data_view(vector, kVectorDistanceOffset + 1);
          const Real dist_z = vector_data_view(vector, kVectorDistanceOffset + 2);
          const Real dist_sqr = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;

          Real time_fraction = static_cast<Real>(0);
          if (dist_sqr > static_cast<Real>(1.0e-24)) {
            const Real vec_sx = vx - vector_data_view(vector, kVectorStartOffset + 0);
            const Real vec_sy = vy - vector_data_view(vector, kVectorStartOffset + 1);
            const Real vec_sz = vz - vector_data_view(vector, kVectorStartOffset + 2);
            const Real dot_dist = vec_sx * dist_x + vec_sy * dist_y + vec_sz * dist_z;
            time_fraction = dot_dist / dist_sqr;
          }

          time_fraction = clamp01(time_fraction);
          const Real start_time = vector_data_view(vector, kVectorStartTimeOffset);
          const Real end_time = vector_data_view(vector, kVectorEndTimeOffset);
          const Real time = start_time + time_fraction * (end_time - start_time);

          const Real vec_path_x =
              vx - (vector_data_view(vector, kVectorStartOffset + 0) + time_fraction * dist_x);
          const Real vec_path_y =
              vy - (vector_data_view(vector, kVectorStartOffset + 1) + time_fraction * dist_y);
          const Real vec_path_z =
              vz - (vector_data_view(vector, kVectorStartOffset + 2) + time_fraction * dist_z);

          const Real local_y = vec_path_x * vector_data_view(vector, kVectorE0Offset + 0) +
                               vec_path_y * vector_data_view(vector, kVectorE0Offset + 1) +
                               vec_path_z * vector_data_view(vector, kVectorE0Offset + 2);
          const Real local_z = vec_path_x * vector_data_view(vector, kVectorE2Offset + 0) +
                               vec_path_y * vector_data_view(vector, kVectorE2Offset + 1) +
                               vec_path_z * vector_data_view(vector, kVectorE2Offset + 2);

          Real width = static_cast<Real>(0);
          Real depth = static_cast<Real>(0);
          Real height = static_cast<Real>(0);
          const Real two_pi_t =
              static_cast<Real>(2.0 * 3.14159265358979323846) * time;
          for (int mode = 0; mode < n_modes; ++mode) {
            const Real phase = phases_view(vector, mode);
            width += mode_data_view(mode, kModeWidthAmplitudeOffset) *
                     fastCos(two_pi_t * mode_data_view(mode, kModeWidthFrequencyOffset) + phase);
            depth += mode_data_view(mode, kModeDepthAmplitudeOffset) *
                     fastCos(two_pi_t * mode_data_view(mode, kModeDepthFrequencyOffset) + phase);
            height += mode_data_view(mode, kModeHeightAmplitudeOffset) *
                      fastCos(two_pi_t * mode_data_view(mode, kModeHeightFrequencyOffset) + phase);
          }

          if (isInside<Real>(local_y, local_z, width, height, depth, height_shape_factor,
                             depth_shape_factor)) {
            is_voxel_melted = 1;
            break;
          }
        }

        melted_view(voxel_index) = is_voxel_melted;
      });
}

// Launch a scalar voxel-major kernel that reuses cached vector invariants and row scalars.
template <typename Real, typename VectorView, typename PhaseView, typename ModeView,
          typename MeltedView>
void launchMeltMaskKernelCached(const GridT<Real>& grid, const VectorView& vector_data_view,
                                const PhaseView& phases_view, const ModeView& mode_data_view,
                                int n_voxels, int n_vectors, int n_modes,
                                Real height_shape_factor, Real depth_shape_factor,
                                const MeltedView& melted_view) {
  const int ny = static_cast<int>(grid.shape[1]);
  const int nz = static_cast<int>(grid.shape[2]);
  const Real origin_x = grid.origin[0];
  const Real origin_y = grid.origin[1];
  const Real origin_z = grid.origin[2];
  const Real resolution = grid.resolution;

  Kokkos::parallel_for(
      "raptor::compute_melt_mask", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index) {
        const int z_index = voxel_index % nz;
        const int yz_index = voxel_index / nz;
        const int y_index = yz_index % ny;
        const int x_index = yz_index / ny;
        const Real vx = origin_x + static_cast<Real>(x_index) * resolution;
        const Real vy = origin_y + static_cast<Real>(y_index) * resolution;
        const Real vz = origin_z + static_cast<Real>(z_index) * resolution;
        std::uint8_t lane_melted[kMaxSeedBatchLanes] = {0, 0, 0, 0};

        const auto phase_accessor = [&](int vector, int mode, int lane) {
          (void)lane;
          return phases_view(vector, mode);
        };

        for (int vector = 0; vector < n_vectors; ++vector) {
          const auto vector_value = [&](int offset) { return vector_data_view(vector, offset); };
          if (evaluateVoxelAgainstPackedVector<Real>(
                  vector_value, vector, vx, vy, vz, mode_data_view, n_modes,
                  height_shape_factor, depth_shape_factor, phase_accessor, 1, lane_melted)) {
            break;
          }
        }

        melted_view(voxel_index) = lane_melted[0];
      });
}

// Launch a packed-repeat kernel that stores one integer bitmask per voxel instead of one byte per
// repeat and evaluates all active seed bits after the geometry culling stage succeeds.
template <typename Real, typename MaskType, typename VectorView, typename PhaseView,
          typename ModeView, typename MaskView>
void launchMeltMaskKernelBitpacked(const GridT<Real>& grid, const VectorView& vector_data_view,
                                   const PhaseView& phases_view, const ModeView& mode_data_view,
                                   int n_voxels, int n_vectors, int n_modes,
                                   Real height_shape_factor, Real depth_shape_factor,
                                   bool enable_random_phases, std::uint64_t batch_seed,
                                   int bit_count, const MaskView& mask_view) {
  const int ny = static_cast<int>(grid.shape[1]);
  const int nz = static_cast<int>(grid.shape[2]);
  const Real origin_x = grid.origin[0];
  const Real origin_y = grid.origin[1];
  const Real origin_z = grid.origin[2];
  const Real resolution = grid.resolution;
  const MaskType full_mask = fullMaskForBits<MaskType>(bit_count);

  Kokkos::parallel_for(
      "raptor::compute_melt_mask", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index) {
        constexpr int kMaskBits = static_cast<int>(sizeof(MaskType) * 8);
        const int z_index = voxel_index % nz;
        const int yz_index = voxel_index / nz;
        const int y_index = yz_index % ny;
        const int x_index = yz_index / ny;
        const Real vx = origin_x + static_cast<Real>(x_index) * resolution;
        const Real vy = origin_y + static_cast<Real>(y_index) * resolution;
        const Real vz = origin_z + static_cast<Real>(z_index) * resolution;
        MaskType melted_mask = static_cast<MaskType>(0);
        Real width_accum[kMaskBits];
        Real depth_accum[kMaskBits];
        Real height_accum[kMaskBits];

        for (int vector = 0; vector < n_vectors; ++vector) {
          if (vx < vector_data_view(vector, kVectorAabbOffset + 0) ||
              vx > vector_data_view(vector, kVectorAabbOffset + 1) ||
              vy < vector_data_view(vector, kVectorAabbOffset + 2) ||
              vy > vector_data_view(vector, kVectorAabbOffset + 3) ||
              vz < vector_data_view(vector, kVectorAabbOffset + 4) ||
              vz > vector_data_view(vector, kVectorAabbOffset + 5)) {
            continue;
          }

          const Real vec_cx = vx - vector_data_view(vector, kVectorCentroidOffset + 0);
          const Real vec_cy = vy - vector_data_view(vector, kVectorCentroidOffset + 1);
          const Real vec_cz = vz - vector_data_view(vector, kVectorCentroidOffset + 2);

          const Real dot_e0 = vec_cx * vector_data_view(vector, kVectorE0Offset + 0) +
                              vec_cy * vector_data_view(vector, kVectorE0Offset + 1) +
                              vec_cz * vector_data_view(vector, kVectorE0Offset + 2);
          if ((dot_e0 * dot_e0) > vector_data_view(vector, kVectorL0SquaredOffset)) {
            continue;
          }

          const Real dot_e1 = vec_cx * vector_data_view(vector, kVectorE1Offset + 0) +
                              vec_cy * vector_data_view(vector, kVectorE1Offset + 1) +
                              vec_cz * vector_data_view(vector, kVectorE1Offset + 2);
          if ((dot_e1 * dot_e1) > vector_data_view(vector, kVectorL1SquaredOffset)) {
            continue;
          }

          Real time_fraction = static_cast<Real>(0);
          const Real inv_dist_sqr = vector_data_view(vector, kVectorInvDistSquaredOffset);
          if (inv_dist_sqr > static_cast<Real>(0)) {
            const Real vec_sx = vx - vector_data_view(vector, kVectorStartOffset + 0);
            const Real vec_sy = vy - vector_data_view(vector, kVectorStartOffset + 1);
            const Real vec_sz = vz - vector_data_view(vector, kVectorStartOffset + 2);
            const Real dot_dist = vec_sx * vector_data_view(vector, kVectorDistanceOffset + 0) +
                                  vec_sy * vector_data_view(vector, kVectorDistanceOffset + 1) +
                                  vec_sz * vector_data_view(vector, kVectorDistanceOffset + 2);
            time_fraction = dot_dist * inv_dist_sqr;
          }

          time_fraction = clamp01(time_fraction);
          const Real time = vector_data_view(vector, kVectorStartTimeOffset) +
                            time_fraction * vector_data_view(vector, kVectorDurationOffset);
          const Real vec_path_x =
              vx - (vector_data_view(vector, kVectorStartOffset + 0) +
                    time_fraction * vector_data_view(vector, kVectorDistanceOffset + 0));
          const Real vec_path_y =
              vy - (vector_data_view(vector, kVectorStartOffset + 1) +
                    time_fraction * vector_data_view(vector, kVectorDistanceOffset + 1));
          const Real vec_path_z =
              vz - (vector_data_view(vector, kVectorStartOffset + 2) +
                    time_fraction * vector_data_view(vector, kVectorDistanceOffset + 2));

          const Real local_y = vec_path_x * vector_data_view(vector, kVectorE0Offset + 0) +
                               vec_path_y * vector_data_view(vector, kVectorE0Offset + 1) +
                               vec_path_z * vector_data_view(vector, kVectorE0Offset + 2);
          const Real local_z = vec_path_x * vector_data_view(vector, kVectorE2Offset + 0) +
                               vec_path_y * vector_data_view(vector, kVectorE2Offset + 1) +
                               vec_path_z * vector_data_view(vector, kVectorE2Offset + 2);

          for (int bit = 0; bit < bit_count; ++bit) {
            width_accum[bit] = static_cast<Real>(0);
            depth_accum[bit] = static_cast<Real>(0);
            height_accum[bit] = static_cast<Real>(0);
          }

          const MaskType remaining_mask = static_cast<MaskType>(full_mask & ~melted_mask);
          if (remaining_mask == static_cast<MaskType>(0)) {
            break;
          }

          const bool use_random_phase = enable_random_phases && n_modes > 1;
          for (int bit = 0; bit < bit_count; ++bit) {
            const MaskType repeat_mask = static_cast<MaskType>(static_cast<MaskType>(1) << bit);
            if ((remaining_mask & repeat_mask) == static_cast<MaskType>(0)) {
              width_accum[bit] = static_cast<Real>(0);
              depth_accum[bit] = static_cast<Real>(0);
              height_accum[bit] = static_cast<Real>(0);
            }
          }

          for (int mode = 0; mode < n_modes; ++mode) {
            const Real width_amplitude = mode_data_view(mode, kModeWidthAmplitudeOffset);
            const Real depth_amplitude = mode_data_view(mode, kModeDepthAmplitudeOffset);
            const Real height_amplitude = mode_data_view(mode, kModeHeightAmplitudeOffset);
            const Real width_angle = time * mode_data_view(mode, kModeWidthTwoPiFrequencyOffset);
            const Real depth_angle = time * mode_data_view(mode, kModeDepthTwoPiFrequencyOffset);
            const Real height_angle = time * mode_data_view(mode, kModeHeightTwoPiFrequencyOffset);
            Real width_sin = static_cast<Real>(0);
            Real width_cos = static_cast<Real>(0);
            Real depth_sin = static_cast<Real>(0);
            Real depth_cos = static_cast<Real>(0);
            Real height_sin = static_cast<Real>(0);
            Real height_cos = static_cast<Real>(0);
            fastSinCos(width_angle, width_sin, width_cos);
            fastSinCos(depth_angle, depth_sin, depth_cos);
            fastSinCos(height_angle, height_sin, height_cos);

            for (int bit = 0; bit < bit_count; ++bit) {
              const MaskType repeat_mask =
                  static_cast<MaskType>(static_cast<MaskType>(1) << bit);
              if ((remaining_mask & repeat_mask) == static_cast<MaskType>(0)) {
                continue;
              }

              const Real phase =
                  use_random_phase
                      ? (mode == 0 ? static_cast<Real>(0)
                                   : phaseFromSeed<Real>(batch_seed +
                                                             static_cast<std::uint64_t>(bit),
                                                         vector, mode))
                      : phases_view(vector, mode);
              Real phase_sin = static_cast<Real>(0);
              Real phase_cos = static_cast<Real>(0);
              fastSinCos(phase, phase_sin, phase_cos);
              width_accum[bit] +=
                  width_amplitude * (width_cos * phase_cos - width_sin * phase_sin);
              depth_accum[bit] +=
                  depth_amplitude * (depth_cos * phase_cos - depth_sin * phase_sin);
              height_accum[bit] +=
                  height_amplitude * (height_cos * phase_cos - height_sin * phase_sin);
            }
          }

          for (int bit = 0; bit < bit_count; ++bit) {
            const MaskType repeat_mask =
                static_cast<MaskType>(static_cast<MaskType>(1) << bit);
            if ((remaining_mask & repeat_mask) == static_cast<MaskType>(0)) {
              continue;
            }

            if (isInside<Real>(local_y, local_z, width_accum[bit], height_accum[bit],
                               depth_accum[bit], height_shape_factor,
                               depth_shape_factor)) {
              melted_mask = static_cast<MaskType>(melted_mask | repeat_mask);
            }
          }

          if (melted_mask == full_mask) {
            break;
          }
        }

        mask_view(voxel_index) = melted_mask;
      });
}

// Hold the full team-bitpacked kernel body in a named functor so Kokkos can query bounds.
template <typename Real, typename VectorView, typename PhaseView, typename ModeView,
          typename MaskView>
struct TeamBitpackedMeltMaskFunctor {
  using TeamPolicy = Kokkos::TeamPolicy<>;
  using TeamMember = TeamPolicy::member_type;
  using ScratchMemorySpace = typename TeamMember::scratch_memory_space;
  using ScratchWordView =
      Kokkos::View<std::uint64_t*, ScratchMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  VectorView vector_data_view;
  PhaseView phases_view;
  ModeView mode_data_view;
  MaskView mask_view;
  int ny = 0;
  int nz = 0;
  int n_vectors = 0;
  int n_modes = 0;
  Real origin_x = static_cast<Real>(0);
  Real origin_y = static_cast<Real>(0);
  Real origin_z = static_cast<Real>(0);
  Real resolution = static_cast<Real>(0);
  Real height_shape_factor = static_cast<Real>(0);
  Real depth_shape_factor = static_cast<Real>(0);
  bool enable_random_phases = false;
  std::uint64_t base_seed = 0;
  int team_size = 1;

  // Report the dynamic scratch footprint so Kokkos can size and validate the team launch.
  KOKKOS_INLINE_FUNCTION std::size_t team_shmem_size(int requested_team_size) const {
    return ScratchWordView::shmem_size(packedMaskWordCount(requested_team_size));
  }

  // Evaluate one voxel per team and emit one packed 64-bit word array for that voxel.
  KOKKOS_INLINE_FUNCTION void operator()(const TeamMember& team) const {
    ScratchWordView word_masks(team.team_scratch(0), packedMaskWordCount(team_size));

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, packedMaskWordCount(team_size)),
                         [&](const int word_index) { word_masks(word_index) = 0ULL; });
    team.team_barrier();

    const int voxel_index = team.league_rank();
    const int lane = team.team_rank();
    const int z_index = voxel_index % nz;
    const int yz_index = voxel_index / nz;
    const int y_index = yz_index % ny;
    const int x_index = yz_index / ny;
    const Real vx = origin_x + static_cast<Real>(x_index) * resolution;
    const Real vy = origin_y + static_cast<Real>(y_index) * resolution;
    const Real vz = origin_z + static_cast<Real>(z_index) * resolution;
    const std::uint64_t lane_seed = base_seed + static_cast<std::uint64_t>(lane);

    bool lane_melted = false;
    for (int vector = 0; vector < n_vectors && !lane_melted; ++vector) {
      if (vx < vector_data_view(vector, kVectorAabbOffset + 0) ||
          vx > vector_data_view(vector, kVectorAabbOffset + 1) ||
          vy < vector_data_view(vector, kVectorAabbOffset + 2) ||
          vy > vector_data_view(vector, kVectorAabbOffset + 3) ||
          vz < vector_data_view(vector, kVectorAabbOffset + 4) ||
          vz > vector_data_view(vector, kVectorAabbOffset + 5)) {
        continue;
      }

      const Real vec_cx = vx - vector_data_view(vector, kVectorCentroidOffset + 0);
      const Real vec_cy = vy - vector_data_view(vector, kVectorCentroidOffset + 1);
      const Real vec_cz = vz - vector_data_view(vector, kVectorCentroidOffset + 2);

      const Real dot_e0 = vec_cx * vector_data_view(vector, kVectorE0Offset + 0) +
                          vec_cy * vector_data_view(vector, kVectorE0Offset + 1) +
                          vec_cz * vector_data_view(vector, kVectorE0Offset + 2);
      if ((dot_e0 * dot_e0) > vector_data_view(vector, kVectorL0SquaredOffset)) {
        continue;
      }

      const Real dot_e1 = vec_cx * vector_data_view(vector, kVectorE1Offset + 0) +
                          vec_cy * vector_data_view(vector, kVectorE1Offset + 1) +
                          vec_cz * vector_data_view(vector, kVectorE1Offset + 2);
      if ((dot_e1 * dot_e1) > vector_data_view(vector, kVectorL1SquaredOffset)) {
        continue;
      }

      Real time_fraction = static_cast<Real>(0);
      const Real inv_dist_sqr = vector_data_view(vector, kVectorInvDistSquaredOffset);
      if (inv_dist_sqr > static_cast<Real>(0)) {
        const Real vec_sx = vx - vector_data_view(vector, kVectorStartOffset + 0);
        const Real vec_sy = vy - vector_data_view(vector, kVectorStartOffset + 1);
        const Real vec_sz = vz - vector_data_view(vector, kVectorStartOffset + 2);
        const Real dot_dist = vec_sx * vector_data_view(vector, kVectorDistanceOffset + 0) +
                              vec_sy * vector_data_view(vector, kVectorDistanceOffset + 1) +
                              vec_sz * vector_data_view(vector, kVectorDistanceOffset + 2);
        time_fraction = dot_dist * inv_dist_sqr;
      }

      time_fraction = clamp01(time_fraction);
      const Real time = vector_data_view(vector, kVectorStartTimeOffset) +
                        time_fraction * vector_data_view(vector, kVectorDurationOffset);
      const Real vec_path_x = vx - (vector_data_view(vector, kVectorStartOffset + 0) +
                                    time_fraction * vector_data_view(vector, kVectorDistanceOffset + 0));
      const Real vec_path_y = vy - (vector_data_view(vector, kVectorStartOffset + 1) +
                                    time_fraction * vector_data_view(vector, kVectorDistanceOffset + 1));
      const Real vec_path_z = vz - (vector_data_view(vector, kVectorStartOffset + 2) +
                                    time_fraction * vector_data_view(vector, kVectorDistanceOffset + 2));

      const Real local_y = vec_path_x * vector_data_view(vector, kVectorE0Offset + 0) +
                           vec_path_y * vector_data_view(vector, kVectorE0Offset + 1) +
                           vec_path_z * vector_data_view(vector, kVectorE0Offset + 2);
      const Real local_z = vec_path_x * vector_data_view(vector, kVectorE2Offset + 0) +
                           vec_path_y * vector_data_view(vector, kVectorE2Offset + 1) +
                           vec_path_z * vector_data_view(vector, kVectorE2Offset + 2);

      Real width = static_cast<Real>(0);
      Real depth = static_cast<Real>(0);
      Real height = static_cast<Real>(0);
      const bool use_random_phase = enable_random_phases && n_modes > 1;
      for (int mode = 0; mode < n_modes; ++mode) {
        const Real width_amplitude = mode_data_view(mode, kModeWidthAmplitudeOffset);
        const Real depth_amplitude = mode_data_view(mode, kModeDepthAmplitudeOffset);
        const Real height_amplitude = mode_data_view(mode, kModeHeightAmplitudeOffset);
        const Real width_angle = time * mode_data_view(mode, kModeWidthTwoPiFrequencyOffset);
        const Real depth_angle = time * mode_data_view(mode, kModeDepthTwoPiFrequencyOffset);
        const Real height_angle = time * mode_data_view(mode, kModeHeightTwoPiFrequencyOffset);
        Real width_sin = static_cast<Real>(0);
        Real width_cos = static_cast<Real>(0);
        Real depth_sin = static_cast<Real>(0);
        Real depth_cos = static_cast<Real>(0);
        Real height_sin = static_cast<Real>(0);
        Real height_cos = static_cast<Real>(0);
        fastSinCos(width_angle, width_sin, width_cos);
        fastSinCos(depth_angle, depth_sin, depth_cos);
        fastSinCos(height_angle, height_sin, height_cos);

        const Real phase = use_random_phase
                               ? (mode == 0 ? static_cast<Real>(0)
                                            : phaseFromSeed<Real>(lane_seed, vector, mode))
                               : phases_view(vector, mode);
        Real phase_sin = static_cast<Real>(0);
        Real phase_cos = static_cast<Real>(0);
        fastSinCos(phase, phase_sin, phase_cos);
        width += width_amplitude * (width_cos * phase_cos - width_sin * phase_sin);
        depth += depth_amplitude * (depth_cos * phase_cos - depth_sin * phase_sin);
        height += height_amplitude * (height_cos * phase_cos - height_sin * phase_sin);
      }

      lane_melted = isInside<Real>(local_y, local_z, width, height, depth,
                                   height_shape_factor, depth_shape_factor);
    }

    if (lane_melted) {
      const int word_index = lane / kPackedMaskWordBits;
      const int word_bit = lane % kPackedMaskWordBits;
      Kokkos::atomic_or(&word_masks(word_index), static_cast<std::uint64_t>(1) << word_bit);
    }
    team.team_barrier();

    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      for (int word_index = 0; word_index < packedMaskWordCount(team_size); ++word_index) {
        mask_view(voxel_index, word_index) = word_masks(word_index);
      }
    });
  }
};

// Query the team-size bounds Kokkos reports for the team-bitpacked kernel on this backend.
template <typename Real>
TeamBitpackedSizeBounds queryTeamBitpackedKernelSizeBounds() {
  using DeviceView = Kokkos::View<Real**, Kokkos::LayoutRight>;
  using MaskView = Kokkos::View<std::uint64_t**, Kokkos::LayoutRight>;
  const TeamBitpackedMeltMaskFunctor<Real, DeviceView, DeviceView, DeviceView, MaskView> functor{
      DeviceView{}, DeviceView{}, DeviceView{}, MaskView{}, 1, 1, 0, 0,
      static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(0),
      static_cast<Real>(1), static_cast<Real>(0), static_cast<Real>(0), false, 0, 1};
  const Kokkos::TeamPolicy<> policy(1, Kokkos::AUTO);
  return {policy.team_size_recommended(functor, Kokkos::ParallelForTag{}),
          policy.team_size_max(functor, Kokkos::ParallelForTag{})};
}

// Launch the team-based packed-repeat kernel using the named functor queried by the benchmark.
template <typename Real, typename VectorView, typename PhaseView, typename ModeView,
          typename MaskView>
void launchMeltMaskKernelTeamBitpacked(
    const GridT<Real>& grid, const VectorView& vector_data_view, const PhaseView& phases_view,
    const ModeView& mode_data_view, int n_voxels, int n_vectors, int n_modes,
    Real height_shape_factor, Real depth_shape_factor, bool enable_random_phases,
    std::uint64_t base_seed, int team_size, const MaskView& mask_view) {
  using Functor = TeamBitpackedMeltMaskFunctor<Real, VectorView, PhaseView, ModeView, MaskView>;
  const Functor functor{vector_data_view,
                        phases_view,
                        mode_data_view,
                        mask_view,
                        static_cast<int>(grid.shape[1]),
                        static_cast<int>(grid.shape[2]),
                        n_vectors,
                        n_modes,
                        grid.origin[0],
                        grid.origin[1],
                        grid.origin[2],
                        grid.resolution,
                        height_shape_factor,
                        depth_shape_factor,
                        enable_random_phases,
                        base_seed,
                        team_size};
  Kokkos::parallel_for("raptor::compute_melt_mask", Kokkos::TeamPolicy<>(n_voxels, team_size),
                       functor);
}

// Launch a seed-batched voxel-major kernel that shares geometry work across up to four repeats.
template <typename Real, typename VectorView, typename PhaseView, typename ModeView,
          typename MeltedView>
void launchMeltMaskKernelSeedBatch4(const GridT<Real>& grid, const VectorView& vector_data_view,
                                    const PhaseView& phases_view, const ModeView& mode_data_view,
                                    int n_voxels, int n_vectors, int n_modes,
                                    Real height_shape_factor, Real depth_shape_factor,
                                    bool enable_random_phases, std::uint64_t batch_seed,
                                    int lane_count, const MeltedView& melted_view) {
  const int ny = static_cast<int>(grid.shape[1]);
  const int nz = static_cast<int>(grid.shape[2]);
  const Real origin_x = grid.origin[0];
  const Real origin_y = grid.origin[1];
  const Real origin_z = grid.origin[2];
  const Real resolution = grid.resolution;

  Kokkos::parallel_for(
      "raptor::compute_melt_mask", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index) {
        const int z_index = voxel_index % nz;
        const int yz_index = voxel_index / nz;
        const int y_index = yz_index % ny;
        const int x_index = yz_index / ny;
        const Real vx = origin_x + static_cast<Real>(x_index) * resolution;
        const Real vy = origin_y + static_cast<Real>(y_index) * resolution;
        const Real vz = origin_z + static_cast<Real>(z_index) * resolution;
        std::uint8_t lane_melted[kMaxSeedBatchLanes] = {0, 0, 0, 0};

        const auto phase_accessor = [&](int vector, int mode, int lane) {
          if (enable_random_phases) {
            return mode == 0 ? static_cast<Real>(0)
                             : phaseFromSeed<Real>(batch_seed + static_cast<std::uint64_t>(lane),
                                                   vector, mode);
          }
          return phases_view(vector, mode);
        };

        for (int vector = 0; vector < n_vectors; ++vector) {
          const auto vector_value = [&](int offset) { return vector_data_view(vector, offset); };
          if (evaluateVoxelAgainstPackedVector<Real>(
                  vector_value, vector, vx, vy, vz, mode_data_view, n_modes,
                  height_shape_factor, depth_shape_factor, phase_accessor, lane_count,
                  lane_melted)) {
            break;
          }
        }

        for (int lane = 0; lane < lane_count; ++lane) {
          melted_view(voxel_index, lane) = lane_melted[lane];
        }
      });
}

// Launch a team-policy kernel that reuses vector chunks across a tile of nearby voxels.
template <typename Real, typename VectorView, typename PhaseView, typename ModeView,
          typename MeltedView>
void launchMeltMaskKernelTeamTileSeedBatch4(
    const GridT<Real>& grid, const VectorView& vector_data_view, const PhaseView& phases_view,
    const ModeView& mode_data_view, int n_voxels, int n_vectors, int n_modes,
    Real height_shape_factor, Real depth_shape_factor, bool enable_random_phases,
    std::uint64_t batch_seed, int lane_count, const MeltedView& melted_view) {
  using TeamPolicy = Kokkos::TeamPolicy<>;
  using TeamMember = TeamPolicy::member_type;
  using ScratchMemorySpace = typename TeamMember::scratch_memory_space;
  using ScratchView =
      Kokkos::View<Real*, ScratchMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const int ny = static_cast<int>(grid.shape[1]);
  const int nz = static_cast<int>(grid.shape[2]);
  const Real origin_x = grid.origin[0];
  const Real origin_y = grid.origin[1];
  const Real origin_z = grid.origin[2];
  const Real resolution = grid.resolution;
  const int tile_voxels = kTeamTileVoxels;
  const int vector_chunk_width = kTeamVectorChunk;
  const int tile_count = (n_voxels + tile_voxels - 1) / tile_voxels;
  const std::size_t scratch_bytes =
      ScratchView::shmem_size(vector_chunk_width * kVectorColumns);

  TeamPolicy policy(tile_count, tile_voxels);
  policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

  Kokkos::parallel_for(
      "raptor::compute_melt_mask", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        ScratchView scratch(team.team_scratch(0), vector_chunk_width * kVectorColumns);

        const int local_voxel = team.team_rank();
        const int voxel_index = team.league_rank() * tile_voxels + local_voxel;
        const bool voxel_active = voxel_index < n_voxels;

        std::uint8_t lane_melted[kMaxSeedBatchLanes] = {0, 0, 0, 0};
        Real vx = static_cast<Real>(0);
        Real vy = static_cast<Real>(0);
        Real vz = static_cast<Real>(0);
        if (voxel_active) {
          const int z_index = voxel_index % nz;
          const int yz_index = voxel_index / nz;
          const int y_index = yz_index % ny;
          const int x_index = yz_index / ny;
          vx = origin_x + static_cast<Real>(x_index) * resolution;
          vy = origin_y + static_cast<Real>(y_index) * resolution;
          vz = origin_z + static_cast<Real>(z_index) * resolution;
        }

        const auto phase_accessor = [&](int vector, int mode, int lane) {
          if (enable_random_phases) {
            return mode == 0 ? static_cast<Real>(0)
                             : phaseFromSeed<Real>(batch_seed + static_cast<std::uint64_t>(lane),
                                                   vector, mode);
          }
          return phases_view(vector, mode);
        };

        for (int chunk_begin = 0; chunk_begin < n_vectors; chunk_begin += vector_chunk_width) {
          const int remaining_vectors = n_vectors - chunk_begin;
          const int chunk_count =
              remaining_vectors < vector_chunk_width ? remaining_vectors : vector_chunk_width;

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, chunk_count * kVectorColumns),
              [&](const int index) {
                const int local_vector = index / kVectorColumns;
                const int column = index % kVectorColumns;
                scratch[local_vector * kVectorColumns + column] =
                    vector_data_view(chunk_begin + local_vector, column);
              });
          team.team_barrier();

          if (voxel_active && !allLanesMelted(lane_melted, lane_count)) {
            for (int local_vector = 0; local_vector < chunk_count; ++local_vector) {
              const int vector_index = chunk_begin + local_vector;
              const auto vector_value = [&](int offset) {
                return scratch[local_vector * kVectorColumns + offset];
              };
              if (evaluateVoxelAgainstPackedVector<Real>(
                      vector_value, vector_index, vx, vy, vz, mode_data_view, n_modes,
                      height_shape_factor, depth_shape_factor, phase_accessor, lane_count,
                      lane_melted)) {
                break;
              }
            }
          }
          team.team_barrier();
        }

        if (voxel_active) {
          for (int lane = 0; lane < lane_count; ++lane) {
            melted_view(voxel_index, lane) = lane_melted[lane];
          }
        }
      });
}

// Count melted voxels in a scalar result buffer.
std::size_t countMeltedVoxels(const Kokkos::View<std::uint8_t*>& melted_view, int n_voxels) {
  std::size_t melted_count = 0;
  Kokkos::parallel_reduce(
      "raptor::count_melted_voxels", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index, std::size_t& local_count) {
        local_count += static_cast<std::size_t>(melted_view(voxel_index) != 0);
      },
      melted_count);
  return melted_count;
}

// Count melted voxels for one seed lane in a batched result buffer.
template <typename MeltedView>
std::size_t countMeltedBatchLane(const MeltedView& melted_view, int lane, int n_voxels) {
  std::size_t melted_count = 0;
  Kokkos::parallel_reduce(
      "raptor::count_melted_voxels", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index, std::size_t& local_count) {
        local_count += static_cast<std::size_t>(melted_view(voxel_index, lane) != 0);
      },
      melted_count);
  return melted_count;
}

// Count melted voxels for one repeat bit in a packed repeat-mask buffer.
template <typename MaskView, typename MaskType>
std::size_t countMeltedPackedBit(const MaskView& mask_view, MaskType repeat_mask, int n_voxels) {
  std::size_t melted_count = 0;
  Kokkos::parallel_reduce(
      "raptor::count_melted_voxels", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index, std::size_t& local_count) {
        local_count += static_cast<std::size_t>((mask_view(voxel_index) & repeat_mask) != 0);
      },
      melted_count);
  return melted_count;
}

// Count melted voxels for one repeat bit stored in a multiword packed mask buffer.
template <typename MaskView>
std::size_t countMeltedPackedBit(const MaskView& mask_view, int word_index,
                                 std::uint64_t repeat_mask, int n_voxels) {
  std::size_t melted_count = 0;
  Kokkos::parallel_reduce(
      "raptor::count_melted_voxels", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index, std::size_t& local_count) {
        local_count +=
            static_cast<std::size_t>((mask_view(voxel_index, word_index) & repeat_mask) != 0);
      },
      melted_count);
  return melted_count;
}

// Invert the scalar melted mask into the porosity convention used by the public API.
void invertScalarMaskToPorosity(const Kokkos::View<std::uint8_t*>& melted_view,
                                const Kokkos::View<std::uint8_t*>& porosity_view,
                                int n_voxels) {
  Kokkos::parallel_for(
      "raptor::invert_melt_mask", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index) {
        porosity_view(voxel_index) = static_cast<std::uint8_t>(melted_view(voxel_index) == 0);
      });
}

// Invert one seed lane from a batched melted mask into the porosity convention.
template <typename MeltedView>
void invertBatchMaskLaneToPorosity(const MeltedView& melted_view, int lane,
                                   const Kokkos::View<std::uint8_t*>& porosity_view,
                                   int n_voxels) {
  Kokkos::parallel_for(
      "raptor::invert_melt_mask", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index) {
        porosity_view(voxel_index) =
            static_cast<std::uint8_t>(melted_view(voxel_index, lane) == 0);
      });
}

// Extract one repeat bit from a packed repeat-mask buffer into the porosity convention.
template <typename MaskView, typename MaskType>
void invertPackedMaskBitToPorosity(const MaskView& mask_view, MaskType repeat_mask,
                                   const Kokkos::View<std::uint8_t*>& porosity_view,
                                   int n_voxels) {
  Kokkos::parallel_for(
      "raptor::invert_melt_mask", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index) {
        porosity_view(voxel_index) =
            static_cast<std::uint8_t>((mask_view(voxel_index) & repeat_mask) == 0);
      });
}

// Extract one repeat bit from a multiword packed repeat-mask buffer into the porosity convention.
template <typename MaskView>
void invertPackedMaskBitToPorosity(const MaskView& mask_view, int word_index,
                                   std::uint64_t repeat_mask,
                                   const Kokkos::View<std::uint8_t*>& porosity_view,
                                   int n_voxels) {
  Kokkos::parallel_for(
      "raptor::invert_melt_mask", Kokkos::RangePolicy<>(0, n_voxels),
      KOKKOS_LAMBDA(const int voxel_index) {
        porosity_view(voxel_index) =
            static_cast<std::uint8_t>((mask_view(voxel_index, word_index) & repeat_mask) == 0);
      });
}

// Append one repeat's morphology rows into the accumulated multi-run output table.
void appendMorphologyRows(PorosityRunSummary& summary, MorphologyTable repeat_table,
                          std::size_t repeats, std::size_t repeat_index) {
  if (summary.accumulated_morphology.headers.empty()) {
    summary.accumulated_morphology.headers = repeat_table.headers;
    if (repeats > 1) {
      summary.accumulated_morphology.headers.insert(
          summary.accumulated_morphology.headers.begin(), "repeat");
    }
  }

  for (std::vector<std::string>& row : repeat_table.rows) {
    if (repeats > 1) {
      row.insert(row.begin(), std::to_string(repeat_index + 1));
    }
    summary.accumulated_morphology.rows.push_back(std::move(row));
  }
}

// Run packed repeat batches using one integer mask per voxel and extract per-repeat results only
// when a count, morphology pass, or final host output needs that specific repeat.
template <typename Real, typename MaskType, typename VectorView, typename PhaseView,
          typename ModeView>
void runPackedRepeatBatches(const GridT<Real>& grid, const VectorView& vector_data_view,
                            const PhaseView& phases_view, const ModeView& mode_data_view,
                            int n_voxels, int n_vectors, int n_modes, Real height_shape_factor,
                            Real depth_shape_factor, const MeltPoolT<Real>& melt_pool,
                            std::size_t repeats, std::uint64_t base_seed,
                            bool copy_final_porosity,
                            const std::vector<std::string>& morphology_fields,
                            const Kokkos::View<std::uint8_t*>& porosity_view,
                            PorosityRunSummary& summary) {
  constexpr int kBitsPerMask = static_cast<int>(sizeof(MaskType) * 8);
  Kokkos::View<MaskType*> packed_mask_view("packed_mask", n_voxels);

  for (std::size_t batch_begin = 0; batch_begin < repeats; batch_begin += kBitsPerMask) {
    const int bit_count =
        static_cast<int>(std::min<std::size_t>(kBitsPerMask, repeats - batch_begin));

    {
      Kokkos::Profiling::ScopedRegion clear_region("raptor::porosity_runs::clear_mask");
      Kokkos::deep_copy(packed_mask_view, static_cast<MaskType>(0));
    }

    {
      Kokkos::Profiling::ScopedRegion variant_region("raptor::variant::bitpacked_repeats");
      launchMeltMaskKernelBitpacked<Real, MaskType>(
          grid, vector_data_view, phases_view, mode_data_view, n_voxels, n_vectors, n_modes,
          height_shape_factor, depth_shape_factor,
          melt_pool.enable_random_phases && n_modes > 1, base_seed + batch_begin, bit_count,
          packed_mask_view);
    }

    for (int bit = 0; bit < bit_count; ++bit) {
      const std::size_t repeat = batch_begin + static_cast<std::size_t>(bit);
      const MaskType repeat_mask =
          static_cast<MaskType>(static_cast<MaskType>(1) << bit);

      {
        Kokkos::Profiling::ScopedRegion analyze_region("raptor::porosity_runs::analyze");
        summary.melted_voxel_counts[repeat] =
            countMeltedPackedBit(packed_mask_view, repeat_mask, n_voxels);
      }

      const bool needs_repeat_porosity =
          !morphology_fields.empty() || (copy_final_porosity && repeat + 1 == repeats);
      if (needs_repeat_porosity) {
        Kokkos::Profiling::ScopedRegion invert_region("raptor::porosity_runs::invert_final");
        invertPackedMaskBitToPorosity(packed_mask_view, repeat_mask, porosity_view, n_voxels);
      }

      if (!morphology_fields.empty()) {
        Kokkos::Profiling::ScopedRegion morphology_region(
            "raptor::porosity_runs::compute_morphology");
        appendMorphologyRows(summary,
                             computeMorphologyFromDevice<Real>(porosity_view, grid.shape,
                                                               grid.resolution, morphology_fields),
                             repeats, repeat);
      }
    }
  }
}

// Run the team-based packed kernel with one team member assigned to each repeat in the batch.
template <typename Real, typename VectorView, typename PhaseView, typename ModeView>
void runTeamBitpackedRepeat(const GridT<Real>& grid, const VectorView& vector_data_view,
                            const PhaseView& phases_view, const ModeView& mode_data_view,
                            int n_voxels, int n_vectors, int n_modes,
                            Real height_shape_factor, Real depth_shape_factor,
                            const MeltPoolT<Real>& melt_pool, std::uint64_t base_seed,
                            int team_size, const char* variant_region_name,
                            bool copy_final_porosity,
                            const std::vector<std::string>& morphology_fields,
                            const Kokkos::View<std::uint8_t*>& porosity_view,
                            PorosityRunSummary& summary) {
  const int word_count = packedMaskWordCount(team_size);
  Kokkos::View<std::uint64_t**, Kokkos::LayoutRight> packed_mask_view("packed_mask", n_voxels,
                                                                      word_count);

  {
    Kokkos::Profiling::ScopedRegion clear_region("raptor::porosity_runs::clear_mask");
    Kokkos::deep_copy(packed_mask_view, static_cast<std::uint64_t>(0));
  }

  {
    Kokkos::Profiling::ScopedRegion variant_region(variant_region_name);
    launchMeltMaskKernelTeamBitpacked<Real>(
        grid, vector_data_view, phases_view, mode_data_view, n_voxels, n_vectors, n_modes,
        height_shape_factor, depth_shape_factor, melt_pool.enable_random_phases && n_modes > 1,
        base_seed, team_size, packed_mask_view);
  }

  for (int bit = 0; bit < team_size; ++bit) {
    const std::size_t repeat = static_cast<std::size_t>(bit);
    const int word_index = bit / kPackedMaskWordBits;
    const int word_bit = bit % kPackedMaskWordBits;
    const std::uint64_t repeat_mask = static_cast<std::uint64_t>(1) << word_bit;

    {
      Kokkos::Profiling::ScopedRegion analyze_region("raptor::porosity_runs::analyze");
      summary.melted_voxel_counts[repeat] =
          countMeltedPackedBit(packed_mask_view, word_index, repeat_mask, n_voxels);
    }

    const bool needs_repeat_porosity =
        !morphology_fields.empty() || (copy_final_porosity && repeat + 1 == summary.melted_voxel_counts.size());
    if (needs_repeat_porosity) {
      Kokkos::Profiling::ScopedRegion invert_region("raptor::porosity_runs::invert_final");
      invertPackedMaskBitToPorosity(packed_mask_view, word_index, repeat_mask, porosity_view,
                                    n_voxels);
    }

    if (!morphology_fields.empty()) {
      Kokkos::Profiling::ScopedRegion morphology_region(
          "raptor::porosity_runs::compute_morphology");
      appendMorphologyRows(summary,
                           computeMorphologyFromDevice<Real>(porosity_view, grid.shape,
                                                             grid.resolution, morphology_fields),
                           summary.melted_voxel_counts.size(), repeat);
    }
  }
}

// Validate the explicit team-size sweep used by the dedicated team benchmark app.
template <typename Real>
void validateTeamBitpackedSize(std::size_t repeats, int team_size) {
  const TeamBitpackedSizeBounds bounds = queryTeamBitpackedKernelSizeBounds<Real>();
  if (team_size <= 0 || team_size > bounds.maximum) {
    throw std::invalid_argument("team bitpacked benchmark team size exceeds the Kokkos-reported maximum.");
  }
  if ((team_size & (team_size - 1)) != 0) {
    throw std::invalid_argument("team bitpacked benchmark requires a power-of-two team size.");
  }
  if (repeats != static_cast<std::size_t>(team_size)) {
    throw std::invalid_argument(
        "team bitpacked benchmark requires repeats to equal the team size.");
  }
}

// Evaluate the asymmetric Lamé cross-section used by the Python kernel.
template <typename Real>
KOKKOS_INLINE_FUNCTION bool isInside(Real y, Real z, Real width, Real height, Real depth,
                                     Real height_shape_factor, Real depth_shape_factor) {
  const Real a = width / static_cast<Real>(2);
  const Real b = z >= static_cast<Real>(0) ? height : depth;
  const Real n = z >= static_cast<Real>(0) ? height_shape_factor : depth_shape_factor;
  if (a <= static_cast<Real>(0) || b <= static_cast<Real>(0)) {
    return false;
  }
  const Real test_value =
      (y / a) * (y / a) + Kokkos::pow(Kokkos::abs(z) / b, n);
  return test_value <= static_cast<Real>(1);
}

}  // namespace

template <typename Real>
std::vector<std::uint8_t> computeMeltMask(const GridT<Real>& grid,
                                          const MeltPoolT<Real>& melt_pool,
                                          const std::vector<PathVectorT<Real>>& path_vectors) {
  Kokkos::Profiling::ScopedRegion region("raptor::kokkos_kernel_setup");

  // Short-circuit the empty-vector case instead of launching a degenerate kernel.
  if (path_vectors.empty() || grid.n_voxels == 0 || melt_pool.n_modes == 0) {
    return std::vector<std::uint8_t>(grid.n_voxels, 0);
  }

  // Size the packed staging buffers from the public inputs.
  const int n_voxels = static_cast<int>(grid.n_voxels);
  const int n_vectors = static_cast<int>(path_vectors.size());
  const int n_modes = static_cast<int>(melt_pool.n_modes);

  // Stage the per-vector and melt-pool data in compact host buffers.
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace> vector_data_host(
      "vector_data_host", n_vectors, kVectorColumns);
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace> phases_host(
      "phases_host", n_vectors, n_modes);
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace> mode_data_host(
      "mode_data_host", n_modes, kModeColumns);

  // Pack the stable vector geometry and prepared phases once for the device kernel.
  for (int vector = 0; vector < n_vectors; ++vector) {
    packStaticVectorRow<Real>(vector_data_host, vector, path_vectors[vector], melt_pool);
  }
  packPreparedPhases<Real>(phases_host, path_vectors, melt_pool, n_vectors, n_modes);
  packModeData<Real>(mode_data_host, melt_pool, n_modes);

  // Allocate the device buffers that will receive the packed host data.
  Kokkos::View<Real**, Kokkos::LayoutRight> vector_data_view("vector_data", n_vectors,
                                                             kVectorColumns);
  Kokkos::View<Real**, Kokkos::LayoutRight> phases_view("phases", n_vectors, n_modes);
  Kokkos::View<Real**, Kokkos::LayoutRight> mode_data_view("mode_data", n_modes, kModeColumns);
  Kokkos::View<std::uint8_t*> melted_view("melted", n_voxels);

  // Move each packed staging buffer to device memory in one shot.
  {
    Kokkos::Profiling::ScopedRegion copy_region("raptor::melt_mask::deep_copy");
    Kokkos::deep_copy(vector_data_view, vector_data_host);
    Kokkos::deep_copy(phases_view, phases_host);
    Kokkos::deep_copy(mode_data_view, mode_data_host);
    Kokkos::deep_copy(melted_view, static_cast<std::uint8_t>(0));
  }

  const Real height_shape_factor = melt_pool.height_shape_factor;
  const Real depth_shape_factor = melt_pool.depth_shape_factor;

  // Launch the preserved baseline kernel for the direct melt-mask API.
  launchMeltMaskKernelBaseline<Real>(grid, vector_data_view, phases_view, mode_data_view,
                                     n_voxels, n_vectors, n_modes, height_shape_factor,
                                     depth_shape_factor, melted_view);

  Kokkos::fence();

  // Read back the device result directly into the returned host vector.
  std::vector<std::uint8_t> melted_mask(grid.n_voxels, 0);
  {
    Kokkos::Profiling::ScopedRegion readback_region("raptor::melt_mask::readback");
    Kokkos::View<std::uint8_t*, Kokkos::LayoutRight, Kokkos::HostSpace> melted_host(
        melted_mask.data(), grid.n_voxels);
    Kokkos::deep_copy(melted_host, melted_view);
  }
  return melted_mask;
}

template <typename Real>
PorosityRunSummary computePorosityRuns(const GridT<Real>& grid,
                                       const std::vector<PathVectorT<Real>>& path_vectors,
                                       const MeltPoolT<Real>& melt_pool, std::size_t repeats,
                                       std::uint64_t base_seed, bool copy_final_porosity,
                                       const std::vector<std::string>& morphology_fields,
                                       PorosityKernelVariant variant) {
  Kokkos::Profiling::ScopedRegion region("raptor::compute_porosity_runs");

  // Validate the repeat count before building any device buffers.
  if (repeats == 0) {
    throw std::invalid_argument("Repeat count must be at least one.");
  }

  const PorosityKernelVariant resolved_variant = variant;
  PorosityRunSummary summary;
  summary.base_seed = base_seed;
  summary.variant_used = resolved_variant;
  summary.melted_voxel_counts.assign(repeats, 0);

  // Preserve the empty-workflow contract while avoiding unnecessary allocations.
  if (path_vectors.empty() || grid.n_voxels == 0 || melt_pool.n_modes == 0) {
    Kokkos::View<std::uint8_t*> porosity_view("porosity", static_cast<int>(grid.n_voxels));
    if (grid.n_voxels > 0) {
      Kokkos::deep_copy(porosity_view, static_cast<std::uint8_t>(1));
    }
    if (!morphology_fields.empty()) {
      summary.accumulated_morphology =
          computeMorphologyFromDevice<Real>(porosity_view, grid.shape, grid.resolution,
                                            morphology_fields);
    }
    if (copy_final_porosity) {
      summary.final_porosity.assign(grid.n_voxels, 1);
    }
    return summary;
  }

  // Size the reusable buffers that stay resident across all repeat runs.
  const int n_voxels = static_cast<int>(grid.n_voxels);
  const int n_vectors = static_cast<int>(path_vectors.size());
  const int n_modes = static_cast<int>(melt_pool.n_modes);
  const bool needs_device_porosity = copy_final_porosity || !morphology_fields.empty();
  const bool use_team_repeat64_variant =
      resolved_variant == PorosityKernelVariant::team_bitpacked_repeat64;
  const bool use_packed_repeat_variant =
      resolved_variant == PorosityKernelVariant::bitpacked_repeats || use_team_repeat64_variant;
  const bool use_batched_variant =
      resolved_variant == PorosityKernelVariant::seed_batch4 ||
      resolved_variant == PorosityKernelVariant::team_tile_seed_batch4;
  const bool use_scalar_variant = !use_packed_repeat_variant && !use_batched_variant;

  // Pack the static vector geometry and melt-pool data once on the host.
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace> vector_data_host(
      "vector_data_host", n_vectors, kVectorColumns);
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace> mode_data_host(
      "mode_data_host", n_modes, kModeColumns);
  for (int vector = 0; vector < n_vectors; ++vector) {
    packStaticVectorRow<Real>(vector_data_host, vector, path_vectors[vector], melt_pool);
  }
  packModeData<Real>(mode_data_host, melt_pool, n_modes);

  // Keep the reusable device buffers resident across all repeat runs.
  Kokkos::View<Real**, Kokkos::LayoutRight> vector_data_view("vector_data", n_vectors,
                                                             kVectorColumns);
  Kokkos::View<Real**, Kokkos::LayoutRight> phases_view("phases", n_vectors, n_modes);
  Kokkos::View<Real**, Kokkos::LayoutRight> mode_data_view("mode_data", n_modes, kModeColumns);
  Kokkos::View<std::uint8_t*> melted_view;
  Kokkos::View<std::uint8_t*> porosity_view;
  Kokkos::View<std::uint8_t**, Kokkos::LayoutRight> melted_batch_view;
  if (use_scalar_variant) {
    melted_view = Kokkos::View<std::uint8_t*>("melted", n_voxels);
  }
  if (needs_device_porosity) {
    porosity_view = Kokkos::View<std::uint8_t*>("porosity", n_voxels);
  }
  if (use_batched_variant) {
    melted_batch_view =
        Kokkos::View<std::uint8_t**, Kokkos::LayoutRight>("melted_batch", n_voxels,
                                                          kMaxSeedBatchLanes);
  }

  // Seed the static device buffers once before entering the repeat loop.
  {
    Kokkos::Profiling::ScopedRegion copy_region("raptor::porosity_runs::initial_copy");
    Kokkos::deep_copy(vector_data_view, vector_data_host);
    Kokkos::deep_copy(mode_data_view, mode_data_host);
    if (!melt_pool.enable_random_phases || n_modes <= 1) {
      Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace> phases_host("phases_host",
                                                                                n_vectors,
                                                                                n_modes);
      for (int vector = 0; vector < n_vectors; ++vector) {
        for (int mode = 0; mode < n_modes; ++mode) {
          phases_host(vector, mode) = melt_pool.width_oscillations[mode].phase;
        }
      }
      Kokkos::deep_copy(phases_view, phases_host);
    }
  }

  const Real height_shape_factor = melt_pool.height_shape_factor;
  const Real depth_shape_factor = melt_pool.depth_shape_factor;

  // Re-run the device workflow while varying only the seed-dependent phase initialization.
  if (use_packed_repeat_variant) {
    if (use_team_repeat64_variant) {
      if (repeats != static_cast<std::size_t>(kTeamRepeat64Lanes)) {
        throw std::invalid_argument(
            "team_bitpacked_repeat64 requires repeats to be exactly 64.");
      }
      runTeamBitpackedRepeat<Real>(grid, vector_data_view, phases_view, mode_data_view, n_voxels,
                                   n_vectors, n_modes, height_shape_factor, depth_shape_factor,
                                   melt_pool, base_seed, kTeamRepeat64Lanes,
                                   "raptor::variant::team_bitpacked_repeat64",
                                   copy_final_porosity, morphology_fields, porosity_view,
                                   summary);
    } else if (repeats <= 8) {
      runPackedRepeatBatches<Real, std::uint8_t>(
          grid, vector_data_view, phases_view, mode_data_view, n_voxels, n_vectors, n_modes,
          height_shape_factor, depth_shape_factor, melt_pool, repeats, base_seed,
          copy_final_porosity, morphology_fields, porosity_view, summary);
    } else if (repeats <= 16) {
      runPackedRepeatBatches<Real, std::uint16_t>(
          grid, vector_data_view, phases_view, mode_data_view, n_voxels, n_vectors, n_modes,
          height_shape_factor, depth_shape_factor, melt_pool, repeats, base_seed,
          copy_final_porosity, morphology_fields, porosity_view, summary);
    } else if (repeats <= 32) {
      runPackedRepeatBatches<Real, std::uint32_t>(
          grid, vector_data_view, phases_view, mode_data_view, n_voxels, n_vectors, n_modes,
          height_shape_factor, depth_shape_factor, melt_pool, repeats, base_seed,
          copy_final_porosity, morphology_fields, porosity_view, summary);
    } else {
      runPackedRepeatBatches<Real, std::uint64_t>(
          grid, vector_data_view, phases_view, mode_data_view, n_voxels, n_vectors, n_modes,
          height_shape_factor, depth_shape_factor, melt_pool, repeats, base_seed,
          copy_final_porosity, morphology_fields, porosity_view, summary);
    }
  } else if (!use_batched_variant) {
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
      if (melt_pool.enable_random_phases && n_modes > 1) {
        Kokkos::Profiling::ScopedRegion phase_region("raptor::porosity_runs::prepare_phases");
        fillRandomPhases<Real>(phases_view, n_vectors, n_modes, base_seed + repeat);
      }

      {
        Kokkos::Profiling::ScopedRegion clear_region("raptor::porosity_runs::clear_mask");
        Kokkos::deep_copy(melted_view, static_cast<std::uint8_t>(0));
      }

      if (resolved_variant == PorosityKernelVariant::cached) {
        Kokkos::Profiling::ScopedRegion variant_region("raptor::variant::cached");
        launchMeltMaskKernelCached<Real>(grid, vector_data_view, phases_view, mode_data_view,
                                         n_voxels, n_vectors, n_modes, height_shape_factor,
                                         depth_shape_factor, melted_view);
      } else {
        Kokkos::Profiling::ScopedRegion variant_region("raptor::variant::baseline");
        launchMeltMaskKernelBaseline<Real>(grid, vector_data_view, phases_view, mode_data_view,
                                           n_voxels, n_vectors, n_modes, height_shape_factor,
                                           depth_shape_factor, melted_view);
      }

      {
        Kokkos::Profiling::ScopedRegion analyze_region("raptor::porosity_runs::analyze");
        summary.melted_voxel_counts[repeat] = countMeltedVoxels(melted_view, n_voxels);
      }

      const bool needs_repeat_porosity =
          !morphology_fields.empty() || (copy_final_porosity && repeat + 1 == repeats);
      if (needs_repeat_porosity) {
        Kokkos::Profiling::ScopedRegion invert_region("raptor::porosity_runs::invert_final");
        invertScalarMaskToPorosity(melted_view, porosity_view, n_voxels);
      }

      if (!morphology_fields.empty()) {
        Kokkos::Profiling::ScopedRegion morphology_region(
            "raptor::porosity_runs::compute_morphology");
        appendMorphologyRows(summary,
                             computeMorphologyFromDevice<Real>(porosity_view, grid.shape,
                                                               grid.resolution, morphology_fields),
                             repeats, repeat);
      }
    }
  } else {
    for (std::size_t batch_begin = 0; batch_begin < repeats; batch_begin += kMaxSeedBatchLanes) {
      const int lane_count =
          static_cast<int>(std::min<std::size_t>(kMaxSeedBatchLanes, repeats - batch_begin));

      {
        Kokkos::Profiling::ScopedRegion clear_region("raptor::porosity_runs::clear_mask");
        Kokkos::deep_copy(melted_batch_view, static_cast<std::uint8_t>(0));
      }

      if (resolved_variant == PorosityKernelVariant::team_tile_seed_batch4) {
        Kokkos::Profiling::ScopedRegion variant_region(
            "raptor::variant::team_tile_seed_batch4");
        launchMeltMaskKernelTeamTileSeedBatch4<Real>(
            grid, vector_data_view, phases_view, mode_data_view, n_voxels, n_vectors, n_modes,
            height_shape_factor, depth_shape_factor,
            melt_pool.enable_random_phases && n_modes > 1, base_seed + batch_begin, lane_count,
            melted_batch_view);
      } else {
        Kokkos::Profiling::ScopedRegion variant_region("raptor::variant::seed_batch4");
        launchMeltMaskKernelSeedBatch4<Real>(
            grid, vector_data_view, phases_view, mode_data_view, n_voxels, n_vectors, n_modes,
            height_shape_factor, depth_shape_factor,
            melt_pool.enable_random_phases && n_modes > 1, base_seed + batch_begin, lane_count,
            melted_batch_view);
      }

      for (int lane = 0; lane < lane_count; ++lane) {
        const std::size_t repeat = batch_begin + static_cast<std::size_t>(lane);
        {
          Kokkos::Profiling::ScopedRegion analyze_region("raptor::porosity_runs::analyze");
          summary.melted_voxel_counts[repeat] =
              countMeltedBatchLane(melted_batch_view, lane, n_voxels);
        }

        const bool needs_repeat_porosity =
            !morphology_fields.empty() || (copy_final_porosity && repeat + 1 == repeats);
        if (needs_repeat_porosity) {
          Kokkos::Profiling::ScopedRegion invert_region("raptor::porosity_runs::invert_final");
          invertBatchMaskLaneToPorosity(melted_batch_view, lane, porosity_view, n_voxels);
        }

        if (!morphology_fields.empty()) {
          Kokkos::Profiling::ScopedRegion morphology_region(
              "raptor::porosity_runs::compute_morphology");
          appendMorphologyRows(summary,
                               computeMorphologyFromDevice<Real>(porosity_view, grid.shape,
                                                                 grid.resolution, morphology_fields),
                               repeats, repeat);
        }
      }
    }
  }

  // Materialize the final porosity field only when an output path requires host access.
  if (copy_final_porosity) {
    summary.final_porosity.resize(grid.n_voxels, 1);
    Kokkos::Profiling::ScopedRegion readback_region("raptor::porosity_runs::readback");
    Kokkos::View<std::uint8_t*, Kokkos::LayoutRight, Kokkos::HostSpace> porosity_host(
        summary.final_porosity.data(), grid.n_voxels);
    Kokkos::deep_copy(porosity_host, porosity_view);
  }

  return summary;
}

template <typename Real>
PorosityRunSummary computePorosityRunsTeamBitpacked(
    const GridT<Real>& grid, const std::vector<PathVectorT<Real>>& path_vectors,
    const MeltPoolT<Real>& melt_pool, std::size_t repeats, std::uint64_t base_seed, int team_size,
    bool copy_final_porosity, const std::vector<std::string>& morphology_fields) {
  Kokkos::Profiling::ScopedRegion region("raptor::compute_porosity_runs_team_bitpacked");
  validateTeamBitpackedSize<Real>(repeats, team_size);

  PorosityRunSummary summary;
  summary.base_seed = base_seed;
  summary.variant_used = PorosityKernelVariant::team_bitpacked_repeat64;
  summary.melted_voxel_counts.assign(repeats, 0);

  if (path_vectors.empty() || grid.n_voxels == 0 || melt_pool.n_modes == 0) {
    Kokkos::View<std::uint8_t*> porosity_view("porosity", static_cast<int>(grid.n_voxels));
    if (grid.n_voxels > 0) {
      Kokkos::deep_copy(porosity_view, static_cast<std::uint8_t>(1));
    }
    if (!morphology_fields.empty()) {
      summary.accumulated_morphology =
          computeMorphologyFromDevice<Real>(porosity_view, grid.shape, grid.resolution,
                                            morphology_fields);
    }
    if (copy_final_porosity) {
      summary.final_porosity.assign(grid.n_voxels, 1);
    }
    return summary;
  }

  const int n_voxels = static_cast<int>(grid.n_voxels);
  const int n_vectors = static_cast<int>(path_vectors.size());
  const int n_modes = static_cast<int>(melt_pool.n_modes);
  const bool needs_device_porosity = copy_final_porosity || !morphology_fields.empty();

  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace> vector_data_host(
      "vector_data_host", n_vectors, kVectorColumns);
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace> mode_data_host(
      "mode_data_host", n_modes, kModeColumns);
  for (int vector = 0; vector < n_vectors; ++vector) {
    packStaticVectorRow<Real>(vector_data_host, vector, path_vectors[vector], melt_pool);
  }
  packModeData<Real>(mode_data_host, melt_pool, n_modes);

  Kokkos::View<Real**, Kokkos::LayoutRight> vector_data_view("vector_data", n_vectors,
                                                             kVectorColumns);
  Kokkos::View<Real**, Kokkos::LayoutRight> phases_view("phases", n_vectors, n_modes);
  Kokkos::View<Real**, Kokkos::LayoutRight> mode_data_view("mode_data", n_modes, kModeColumns);
  Kokkos::View<std::uint8_t*> porosity_view;
  if (needs_device_porosity) {
    porosity_view = Kokkos::View<std::uint8_t*>("porosity", n_voxels);
  }

  {
    Kokkos::Profiling::ScopedRegion copy_region("raptor::porosity_runs::initial_copy");
    Kokkos::deep_copy(vector_data_view, vector_data_host);
    Kokkos::deep_copy(mode_data_view, mode_data_host);
    if (!melt_pool.enable_random_phases || n_modes <= 1) {
      Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace> phases_host("phases_host",
                                                                                n_vectors,
                                                                                n_modes);
      for (int vector = 0; vector < n_vectors; ++vector) {
        for (int mode = 0; mode < n_modes; ++mode) {
          phases_host(vector, mode) = melt_pool.width_oscillations[mode].phase;
        }
      }
      Kokkos::deep_copy(phases_view, phases_host);
    }
  }

  runTeamBitpackedRepeat<Real>(grid, vector_data_view, phases_view, mode_data_view, n_voxels,
                               n_vectors, n_modes, melt_pool.height_shape_factor,
                               melt_pool.depth_shape_factor, melt_pool, base_seed, team_size,
                               "raptor::variant::team_bitpacked_repeat_power2",
                               copy_final_porosity, morphology_fields, porosity_view, summary);

  if (copy_final_porosity) {
    summary.final_porosity.resize(grid.n_voxels, 1);
    Kokkos::Profiling::ScopedRegion readback_region("raptor::porosity_runs::readback");
    Kokkos::View<std::uint8_t*, Kokkos::LayoutRight, Kokkos::HostSpace> porosity_host(
        summary.final_porosity.data(), grid.n_voxels);
    Kokkos::deep_copy(porosity_host, porosity_view);
  }

  return summary;
}

template <typename Real>
TeamBitpackedSizeBounds queryTeamBitpackedSizeBounds() {
  return queryTeamBitpackedKernelSizeBounds<Real>();
}

template std::vector<std::uint8_t> computeMeltMask<float>(
    const GridT<float>&, const MeltPoolT<float>&, const std::vector<PathVectorT<float>>&);
template std::vector<std::uint8_t> computeMeltMask<double>(
    const GridT<double>&, const MeltPoolT<double>&, const std::vector<PathVectorT<double>>&);
template PorosityRunSummary computePorosityRuns<float>(
    const GridT<float>&, const std::vector<PathVectorT<float>>&, const MeltPoolT<float>&,
    std::size_t, std::uint64_t, bool, const std::vector<std::string>&,
    PorosityKernelVariant);
template PorosityRunSummary computePorosityRuns<double>(
    const GridT<double>&, const std::vector<PathVectorT<double>>&, const MeltPoolT<double>&,
    std::size_t, std::uint64_t, bool, const std::vector<std::string>&,
    PorosityKernelVariant);
template PorosityRunSummary computePorosityRunsTeamBitpacked<float>(
    const GridT<float>&, const std::vector<PathVectorT<float>>&, const MeltPoolT<float>&,
    std::size_t, std::uint64_t, int, bool, const std::vector<std::string>&);
template PorosityRunSummary computePorosityRunsTeamBitpacked<double>(
    const GridT<double>&, const std::vector<PathVectorT<double>>&, const MeltPoolT<double>&,
    std::size_t, std::uint64_t, int, bool, const std::vector<std::string>&);
template TeamBitpackedSizeBounds queryTeamBitpackedSizeBounds<float>();
template TeamBitpackedSizeBounds queryTeamBitpackedSizeBounds<double>();

}  // namespace raptor
