#include "raptor/api.hpp"

#include <Kokkos_Core.hpp>

#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::filesystem::path resolvePath(const std::filesystem::path& base,
                                  const std::filesystem::path& candidate) {
  return candidate.is_absolute() ? candidate : base / candidate;
}

// Scope guard for Kokkos profiling regions used by the top-level application stages.
class ProfilingRegionGuard {
 public:
  // Open a profiling region on construction and close it on destruction.
  explicit ProfilingRegionGuard(const char* name) : name_(name) {
    Kokkos::Profiling::pushRegion(name_);
  }

  // Close the profiling region when the guard leaves scope.
  ~ProfilingRegionGuard() { Kokkos::Profiling::popRegion(); }

  ProfilingRegionGuard(const ProfilingRegionGuard&) = delete;
  ProfilingRegionGuard& operator=(const ProfilingRegionGuard&) = delete;

 private:
  const char* name_;
};

template <typename Real>
raptor::Vec3T<Real> castVec3(const raptor::Vec3& input) {
  return {static_cast<Real>(input[0]), static_cast<Real>(input[1]), static_cast<Real>(input[2])};
}

template <typename Real>
raptor::BoundBoxT<Real> castBoundBox(const raptor::BoundBox& input) {
  return {castVec3<Real>(input[0]), castVec3<Real>(input[1])};
}

template <typename Real>
int runMainTyped(const raptor::SimulationConfig& config) {
  ProfilingRegionGuard run_region("raptor::Main::run");
  std::cout << "Loaded config: " << config.config_path << '\n';
  std::cout << "Floating precision: " << raptor::toString(config.parameters.floating_precision)
            << '\n';

  // Read and prepare the scan vectors from each configured path file.
  std::vector<raptor::PathVectorT<Real>> all_vectors;
  {
    ProfilingRegionGuard region("raptor::Main::load_scan_vectors");
    for (const std::filesystem::path& scan_path : config.scan_paths) {
      const std::filesystem::path resolved_scan_path = resolvePath(config.config_dir, scan_path);
      std::vector<raptor::PathVectorT<Real>> vectors = raptor::readScanPath<Real>(resolved_scan_path);
      for (raptor::PathVectorT<Real>& vector : vectors) {
        vector.setCoordinateFrame();
        all_vectors.push_back(vector);
      }
    }
  }

  // Load the melt-pool inputs and preserve the width/depth/height component split.
  std::map<std::string, raptor::MeltPoolComponentInputT<Real>> melt_pool_inputs;
  {
    ProfilingRegionGuard region("raptor::Main::load_melt_pool_inputs");
    for (const auto& [name, component] : config.melt_pool_data) {
      const std::filesystem::path resolved_data_path =
          resolvePath(config.config_dir, component.file_name);
      melt_pool_inputs[name] = {raptor::readData<Real>(resolved_data_path), component.n_modes,
                                static_cast<Real>(component.scale),
                                static_cast<Real>(component.shape)};
    }
  }
  raptor::MeltPoolT<Real> melt_pool;
  {
    ProfilingRegionGuard region("raptor::Main::create_melt_pool");
    melt_pool =
        raptor::createMeltPool<Real>(melt_pool_inputs,
                                     config.parameters.enable_random_segment_phase);
  }

  // Create the grid from the configured RVE or fall back to the scan-vector envelope.
  const std::optional<raptor::BoundBoxT<Real>> bound_box =
      config.has_rve ? std::optional<raptor::BoundBoxT<Real>>(castBoundBox<Real>(config.rve))
                     : std::nullopt;
  raptor::GridT<Real> grid;
  {
    ProfilingRegionGuard region("raptor::Main::create_grid");
    grid = raptor::createGrid<Real>(
        static_cast<Real>(config.parameters.voxel_resolution), bound_box,
        config.has_rve ? nullptr
                       : static_cast<const std::vector<raptor::PathVectorT<Real>>*>(&all_vectors));
  }

  // Run the porosity workflow and only read the final field back if VTI output is requested.
  const bool needs_host_porosity = config.output.vtk_file_name.has_value();
  const std::uint64_t base_seed = config.parameters.random_seed.has_value()
                                      ? *config.parameters.random_seed
                                      : ((static_cast<std::uint64_t>(std::random_device{}()) << 32) ^
                                         static_cast<std::uint64_t>(std::random_device{}()));
  raptor::PorosityRunSummary porosity_runs;
  {
    ProfilingRegionGuard region("raptor::Main::compute_porosity");
    porosity_runs = raptor::computePorosityRuns<Real>(
        grid, all_vectors, melt_pool, config.parameters.repeats, base_seed, needs_host_porosity,
        config.output.morphology_fields, config.parameters.porosity_variant);
  }
  if (config.parameters.repeats > 1) {
    ProfilingRegionGuard region("raptor::Main::report_repeats");
    std::cout << "Completed " << config.parameters.repeats << " GPU porosity runs using base seed "
              << base_seed << " with variant "
              << raptor::toString(porosity_runs.variant_used) << ".\n";
    for (std::size_t repeat = 0; repeat < porosity_runs.melted_voxel_counts.size(); ++repeat) {
      std::cout << "  Repeat " << (repeat + 1) << ": melted "
                << porosity_runs.melted_voxel_counts[repeat] << " of " << grid.n_voxels
                << " voxels.\n";
    }
  }
  if (!config.output.morphology_fields.empty()) {
    if (!config.output.morphology_file_name.has_value()) {
      throw std::invalid_argument("Morphology fields were requested without an output file.");
    }
    {
      ProfilingRegionGuard region("raptor::Main::write_morphology");
      raptor::writeMorphology(porosity_runs.accumulated_morphology,
                              resolvePath(config.config_dir, *config.output.morphology_file_name));
    }
  }
  if (config.output.vtk_file_name.has_value()) {
    ProfilingRegionGuard region("raptor::Main::write_vti");
    raptor::writeVti<Real>(grid.origin, grid.resolution, grid.shape, porosity_runs.final_porosity,
                           resolvePath(config.config_dir, *config.output.vtk_file_name));
  }

  std::cout << "Completed porosity workflow for " << all_vectors.size() << " path vectors.\n";
  return 0;
}

int runMain(const std::string& config_file) {
  raptor::SimulationConfig config;
  {
    ProfilingRegionGuard region("raptor::Main::parse_config");
    config = raptor::parseConfig(config_file);
  }

  switch (config.parameters.floating_precision) {
    case raptor::FloatingPrecision::single_precision:
      return runMainTyped<float>(config);
    case raptor::FloatingPrecision::double_precision:
      return runMainTyped<double>(config);
  }
  throw std::invalid_argument("Unsupported floating precision selection.");
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  try {
    if (argc != 2) {
      std::cerr << "Usage: Main <config.yaml>\n";
      Kokkos::finalize();
      return 1;
    }
    const int result = runMain(argv[1]);
    Kokkos::finalize();
    return result;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
}
