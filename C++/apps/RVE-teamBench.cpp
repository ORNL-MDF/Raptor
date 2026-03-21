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

// Round a positive integer down to the nearest power of two.
int floorPowerOfTwo(int value) {
  int power = 1;
  while ((power << 1) > 0 && (power << 1) <= value) {
    power <<= 1;
  }
  return power;
}

// Read an unsigned integer benchmark override from the environment.
std::uint64_t readEnvUint64(const char* name, std::uint64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  return static_cast<std::uint64_t>(std::stoull(value));
}

// Build the power-of-two team-size sweep from the active backend and Kokkos-reported bounds.
std::vector<int> buildTeamSizes() {
  const std::string execution_space = Kokkos::DefaultExecutionSpace::name();
  const TeamBitpackedSizeBounds bounds = queryTeamBitpackedSizeBounds<float>();

  int min_team_size = 1;
  int max_team_size = floorPowerOfTwo(std::max(1, bounds.maximum));
  if (execution_space == "Cuda" || execution_space == "HIP" || execution_space == "SYCL" ||
      execution_space == "OpenMPTarget") {
    min_team_size = 32;
  }

  std::vector<int> team_sizes;
  for (int team_size = min_team_size; team_size <= max_team_size; team_size <<= 1) {
    team_sizes.push_back(team_size);
  }
  return team_sizes;
}

// Hold the reusable float-precision RVE inputs for the team-size sweep benchmark.
struct TeamBenchCaseData {
  GridT<float> grid;
  std::vector<PathVectorT<float>> path_vectors;
  MeltPoolT<float> melt_pool;
};

// Build the fixed float RVE benchmark inputs once so only the kernel strategy varies.
TeamBenchCaseData buildBenchmarkCase() {
  TeamBenchCaseData data;

  const std::array<float, 3> min_point = {0.0f, 0.0f, 0.0f};
  const std::array<float, 3> max_point = {5.0e-4f, 5.0e-4f, 5.0e-4f};
  const BoundBoxT<float> bound_box = {min_point, max_point};
  {
    Kokkos::Profiling::ScopedRegion region("raptor::RVETeamBench::create_grid");
    data.grid = createGrid<float>(5.0e-6f, std::optional<BoundBoxT<float>>(bound_box), nullptr);
  }

  {
    Kokkos::Profiling::ScopedRegion region("raptor::RVETeamBench::create_path_vectors");
    data.path_vectors = createPathVectors<float>(bound_box, 370.0f, 1.7f, 140.0e-6f, 30.0e-6f,
                                                 67.0f, 5.0e-4f, 0);
  }

  const std::filesystem::path melt_pool_data_path =
      std::filesystem::current_path() / ".." / "data" / "meltPoolData" /
      "ULI_v1700_theta0_widths.txt";
  const std::vector<std::vector<float>> width_data = readData<float>(melt_pool_data_path);
  const std::map<std::string, MeltPoolComponentInputT<float>> melt_pool_dict = {
      {"width", {width_data, 50, 1.0f, 2.0f}},
      {"depth", {width_data, 50, 0.8f, 1.0f}},
      {"height", {width_data, 50, 0.4f, 1.0f}}};
  {
    Kokkos::Profiling::ScopedRegion region("raptor::RVETeamBench::create_melt_pool");
    data.melt_pool = createMeltPool<float>(melt_pool_dict, true);
  }

  return data;
}

// Run one baseline benchmark case at the repeat count that matches the team size under test.
void runBaselineCase(const TeamBenchCaseData& data, int repeats, std::uint64_t base_seed) {
  const std::string region_name =
      "raptor::RVETeamBench::float::baseline::repeat_" + std::to_string(repeats);
  {
    Kokkos::Profiling::ScopedRegion region(region_name.c_str());
    const PorosityRunSummary summary =
        computePorosityRuns<float>(data.grid, data.path_vectors, data.melt_pool, repeats,
                                   base_seed, false, {}, PorosityKernelVariant::baseline);
    std::cout << "Team bench baseline repeats=" << repeats
              << " first_repeat_melted=" << summary.melted_voxel_counts.front() << '\n';
  }
}

// Run one team-bitpacked case at an explicit power-of-two team size.
void runTeamCase(const TeamBenchCaseData& data, int team_size, std::uint64_t base_seed) {
  const std::string region_name =
      "raptor::RVETeamBench::float::team_bitpacked::team_" + std::to_string(team_size);
  {
    Kokkos::Profiling::ScopedRegion region(region_name.c_str());
    const PorosityRunSummary summary = computePorosityRunsTeamBitpacked<float>(
        data.grid, data.path_vectors, data.melt_pool, static_cast<std::size_t>(team_size),
        base_seed, team_size, false);
    std::cout << "Team bench team_size=" << team_size
              << " first_repeat_melted=" << summary.melted_voxel_counts.front() << '\n';
  }
}

// Sweep power-of-two team sizes up to the packed-mask limit and compare against baseline.
int runTeamBench() {
  const std::uint64_t base_seed = readEnvUint64("RAPTOR_RVE_BASE_SEED", 7);
  printExecutionSpace();
  const TeamBitpackedSizeBounds bounds = queryTeamBitpackedSizeBounds<float>();
  const std::vector<int> team_sizes = buildTeamSizes();
  std::cout << "Team bench bounds: recommended=" << bounds.recommended
            << " max=" << bounds.maximum << '\n';
  std::cout << "Team bench sizes:";
  for (const int team_size : team_sizes) {
    std::cout << ' ' << team_size;
  }
  std::cout << '\n';

  const TeamBenchCaseData data = buildBenchmarkCase();
  for (const int team_size : team_sizes) {
    runBaselineCase(data, team_size, base_seed);
    runTeamCase(data, team_size, base_seed);
  }

  return 0;
}

}  // namespace
}  // namespace raptor

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  try {
    const int result = raptor::runTeamBench();
    Kokkos::finalize();
    return result;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
}
