#include "raptor/api.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace raptor {
namespace {

// Report the active Kokkos execution space so benchmark logs show the backend used.
void printExecutionSpace() {
  std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << '\n';
}

// Read an unsigned integer benchmark override from the environment.
std::uint64_t readEnvUint64(const char* name, std::uint64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  return static_cast<std::uint64_t>(std::stoull(value));
}

// Read a size_t benchmark override from the environment.
std::size_t readEnvSizeT(const char* name, std::size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  return static_cast<std::size_t>(std::stoull(value));
}

// Hold the reusable RVE geometry and melt-pool inputs for one floating-point precision.
template <typename Real>
struct BenchmarkCaseData {
  GridT<Real> grid;
  std::vector<PathVectorT<Real>> path_vectors;
  MeltPoolT<Real> melt_pool;
};

// Build the fixed RVE benchmark inputs once per precision so each variant run sees the same data.
template <typename Real>
BenchmarkCaseData<Real> buildBenchmarkCase() {
  BenchmarkCaseData<Real> data;

  // Build the fixed representative-volume grid used by the benchmark executable.
  const std::array<Real, 3> min_point = {static_cast<Real>(0), static_cast<Real>(0),
                                         static_cast<Real>(0)};
  const std::array<Real, 3> max_point = {static_cast<Real>(5.0e-4), static_cast<Real>(5.0e-4),
                                         static_cast<Real>(5.0e-4)};
  const BoundBoxT<Real> bound_box = {min_point, max_point};
  const Real voxel_resolution = static_cast<Real>(5.0e-6);
  {
    Kokkos::Profiling::ScopedRegion region("raptor::RVEBenchmark::create_grid");
    data.grid = createGrid<Real>(voxel_resolution, std::optional<BoundBoxT<Real>>(bound_box),
                                 nullptr);
  }

  // Build the path-vector set once so only the porosity kernels vary across benchmark runs.
  const Real power = static_cast<Real>(370.0);
  const Real scan_speed = static_cast<Real>(1.7);
  const Real hatch_spacing = static_cast<Real>(140e-6);
  const Real layer_height = static_cast<Real>(30e-6);
  const Real rotation_degrees = static_cast<Real>(67.0);
  const Real scan_extension = static_cast<Real>(5.0e-4);
  const int extra_layers = 0;
  {
    Kokkos::Profiling::ScopedRegion region("raptor::RVEBenchmark::create_path_vectors");
    data.path_vectors =
        createPathVectors<Real>(bound_box, power, scan_speed, hatch_spacing, layer_height,
                                rotation_degrees, scan_extension, extra_layers);
  }

  // Build the melt-pool state once from the benchmark width history.
  const std::filesystem::path melt_pool_data_path =
      std::filesystem::current_path() / ".." / "data" / "meltPoolData" /
      "ULI_v1700_theta0_widths.txt";
  const std::vector<std::vector<Real>> width_data = readData<Real>(melt_pool_data_path);
  const std::size_t n_modes = 50;
  const std::map<std::string, MeltPoolComponentInputT<Real>> melt_pool_dict = {
      {"width", {width_data, n_modes, static_cast<Real>(1.0), static_cast<Real>(2.0)}},
      {"depth", {width_data, n_modes, static_cast<Real>(0.8), static_cast<Real>(1.0)}},
      {"height", {width_data, n_modes, static_cast<Real>(0.4), static_cast<Real>(1.0)}}};
  {
    Kokkos::Profiling::ScopedRegion region("raptor::RVEBenchmark::create_melt_pool");
    data.melt_pool = createMeltPool<Real>(melt_pool_dict, true);
  }

  return data;
}

// Run one precision/variant benchmark case with its own profiling region and summary line.
template <typename Real>
void runBenchmarkCase(const BenchmarkCaseData<Real>& data, FloatingPrecision precision,
                      PorosityKernelVariant variant, std::size_t repeats,
                      std::uint64_t base_seed) {
  const std::string region_name = std::string("raptor::RVEBenchmark::") + toString(precision) +
                                  "::" + toString(variant);
  {
    Kokkos::Profiling::ScopedRegion region(region_name.c_str());
    const PorosityRunSummary summary =
        computePorosityRuns<Real>(data.grid, data.path_vectors, data.melt_pool, repeats,
                                  base_seed, false, {}, variant);
    std::cout << "Benchmark case precision=" << toString(precision)
              << " variant=" << toString(summary.variant_used)
              << " repeats=" << repeats << " base_seed=" << base_seed
              << " first_repeat_melted=" << summary.melted_voxel_counts.front() << '\n';
  }
}

// Run the full benchmark matrix in one process so Kokkos profiling can attribute every case.
int runBenchmark() {
  const std::size_t repeats = readEnvSizeT("RAPTOR_RVE_REPEATS", 64);
  const std::uint64_t base_seed = readEnvUint64("RAPTOR_RVE_BASE_SEED", 7);
  if (repeats != 64) {
    throw std::invalid_argument("RVE-benchmark expects RAPTOR_RVE_REPEATS=64.");
  }

  printExecutionSpace();

  const BenchmarkCaseData<float> float_case = buildBenchmarkCase<float>();
  const BenchmarkCaseData<double> double_case = buildBenchmarkCase<double>();
  const std::array<PorosityKernelVariant, 6> variants = {
      PorosityKernelVariant::baseline,           PorosityKernelVariant::cached,
      PorosityKernelVariant::bitpacked_repeats,  PorosityKernelVariant::seed_batch4,
      PorosityKernelVariant::team_tile_seed_batch4,
      PorosityKernelVariant::team_bitpacked_repeat64};

  for (const PorosityKernelVariant variant : variants) {
    runBenchmarkCase<float>(float_case, FloatingPrecision::single_precision, variant, repeats,
                            base_seed);
  }
  for (const PorosityKernelVariant variant : variants) {
    runBenchmarkCase<double>(double_case, FloatingPrecision::double_precision, variant, repeats,
                             base_seed);
  }

  return 0;
}

}  // namespace
}  // namespace raptor

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  try {
    const int result = raptor::runBenchmark();
    Kokkos::finalize();
    return result;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
}
