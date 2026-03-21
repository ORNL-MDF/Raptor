#include "raptor/api.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace raptor {
namespace {

// Report the active Kokkos execution space so benchmark logs show the actual backend in use.
void printExecutionSpace() {
  std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << '\n';
}

// Read a boolean benchmark override from the environment using 0/1 or true/false strings.
bool readEnvBool(const char* name, bool fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }

  const std::string parsed(value);
  if (parsed == "1" || parsed == "true" || parsed == "TRUE") {
    return true;
  }
  if (parsed == "0" || parsed == "false" || parsed == "FALSE") {
    return false;
  }
  throw std::invalid_argument(std::string("Invalid boolean environment value for ") + name);
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

// Read a porosity-kernel selector override from the environment.
PorosityKernelVariant readEnvVariant(const char* name, PorosityKernelVariant fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  return parsePorosityKernelVariant(value);
}

// Read a float-or-double selector override from the environment.
FloatingPrecision readEnvPrecision(const char* name, FloatingPrecision fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  return parseFloatingPrecision(value);
}

template <typename Real>
int runRVETyped(FloatingPrecision precision) {
  // 1. Create voxel grid for the representative volume element (RVE)
  const std::array<Real, 3> min_point = {static_cast<Real>(0), static_cast<Real>(0),
                                         static_cast<Real>(0)};
  const std::array<Real, 3> max_point = {static_cast<Real>(5.0e-4), static_cast<Real>(5.0e-4),
                                         static_cast<Real>(5.0e-4)};
  const BoundBoxT<Real> bound_box = {min_point, max_point};
  const Real voxel_resolution = static_cast<Real>(5.0e-6);

  GridT<Real> grid;
  {
    Kokkos::Profiling::ScopedRegion region("raptor::RVE::create_grid");
    grid = createGrid<Real>(voxel_resolution, std::optional<BoundBoxT<Real>>(bound_box), nullptr);
  }

  // 2. Create path vectors through the representative volume element (RVE)
  const Real power = static_cast<Real>(370.0);
  const Real scan_speed = static_cast<Real>(1.7);
  const Real hatch_spacing = static_cast<Real>(140e-6);
  const Real layer_height = static_cast<Real>(30e-6);
  const Real rotation_degrees = static_cast<Real>(67.0);
  const Real scan_extension = static_cast<Real>(5.0e-4);  // max dimension
  const int extra_layers = 0;

  std::vector<PathVectorT<Real>> path_vectors;
  {
    Kokkos::Profiling::ScopedRegion region("raptor::RVE::create_path_vectors");
    path_vectors = createPathVectors<Real>(bound_box, power, scan_speed, hatch_spacing,
                                           layer_height, rotation_degrees, scan_extension,
                                           extra_layers);
  }

  // 3. Create melt pools given a width sequence
  const std::filesystem::path melt_pool_data_path =
      std::filesystem::current_path() / ".." / "data" / "meltPoolData" /
      "ULI_v1700_theta0_widths.txt";
  const std::vector<std::vector<Real>> width_data = readData<Real>(melt_pool_data_path);
  const std::size_t n_modes = 50;

  // Scale melt pool data by constant factor.
  const Real width_scale = static_cast<Real>(1.0);
  const Real depth_scale = static_cast<Real>(0.8);
  const Real height_scale = static_cast<Real>(0.4);

  // Assign shape to melt pool (1 = parabola, 2 = ellipse).
  const Real width_shape = static_cast<Real>(2.0);
  const Real height_shape = static_cast<Real>(1.0);
  const Real depth_shape = static_cast<Real>(1.0);

  const std::map<std::string, MeltPoolComponentInputT<Real>> melt_pool_dict = {
      {"width", {width_data, n_modes, width_scale, width_shape}},
      {"depth", {width_data, n_modes, depth_scale, depth_shape}},
      {"height", {width_data, n_modes, height_scale, height_shape}}};

  MeltPoolT<Real> melt_pool;
  {
    Kokkos::Profiling::ScopedRegion region("raptor::RVE::create_melt_pool");
    melt_pool = createMeltPool<Real>(melt_pool_dict, true);  // enable_random_phases
  }

  // 4. Run the GPU-resident porosity workflow and accumulate morphology after all repeats.
  const bool write_vti = readEnvBool("RAPTOR_RVE_WRITE_VTI", true);
  const bool write_morphology = readEnvBool("RAPTOR_RVE_WRITE_MORPHOLOGY", true);
  const std::size_t repeats = readEnvSizeT("RAPTOR_RVE_REPEATS", 1);
  const std::uint64_t base_seed = readEnvUint64("RAPTOR_RVE_BASE_SEED", 7);
  const PorosityKernelVariant variant =
      readEnvVariant("RAPTOR_POROSITY_VARIANT", PorosityKernelVariant::baseline);
  const std::vector<std::string> morphology_fields =
      write_morphology ? std::vector<std::string>{"area", "equivalent_diameter_area"}
                       : std::vector<std::string>{};
  PorosityRunSummary run_summary;
  {
    Kokkos::Profiling::ScopedRegion region("raptor::RVE::compute_porosity");
    run_summary = computePorosityRuns<Real>(grid, path_vectors, melt_pool, repeats, base_seed,
                                            write_vti, morphology_fields, variant);
  }

  // 5. Write the accumulated morphology table after all repeats have completed.
  if (write_morphology) {
    Kokkos::Profiling::ScopedRegion region("raptor::RVE::write_morphology");
    const std::filesystem::path morphology_path = "rve_cpp_morphology.csv";
    writeMorphology(run_summary.accumulated_morphology, morphology_path);
  }

  // 6. Write the final VTI only if the example requests it.
  if (write_vti) {
    Kokkos::Profiling::ScopedRegion region("raptor::RVE::write_vti");
    const std::filesystem::path vtk_path = "rve_cpp.vti";
    writeVti<Real>(grid.origin, grid.resolution, grid.shape, run_summary.final_porosity, vtk_path);
  }

  std::cout << "RVE simulation completed successfully.\n";
  std::cout << "Precision: " << toString(precision) << ", variant: "
            << toString(run_summary.variant_used) << ", repeats: " << repeats
            << ", base seed: " << base_seed << ".\n";
  std::cout << "Generated " << path_vectors.size() << " path vectors.\n";
  std::cout << "Found " << run_summary.accumulated_morphology.rows.size() << " defects.\n";
  for (std::size_t repeat = 0; repeat < run_summary.melted_voxel_counts.size(); ++repeat) {
    std::cout << "  Repeat " << (repeat + 1) << ": melted "
              << run_summary.melted_voxel_counts[repeat] << " voxels.\n";
  }

  return 0;
}

int runRVE() {
  const FloatingPrecision precision =
      readEnvPrecision("RAPTOR_REAL_TYPE", FloatingPrecision::double_precision);
  switch (precision) {
    case FloatingPrecision::single_precision:
      return runRVETyped<float>(precision);
    case FloatingPrecision::double_precision:
      return runRVETyped<double>(precision);
  }
  throw std::invalid_argument("Unsupported floating precision selection.");
}

}  // namespace

}  // namespace raptor

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  try {
    raptor::printExecutionSpace();
    const int result = raptor::runRVE();
    Kokkos::finalize();
    return result;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
}
