#pragma once

#include "raptor/types.hpp"

namespace raptor {

struct MeltPoolComponentConfig {
  std::string type;
  std::filesystem::path file_name;
  std::size_t n_modes = 0;
  double scale = 1.0;
  double shape = 2.0;
};

struct SimulationParameters {
  double layer_height = 0.0;
  double voxel_resolution = 0.0;
  bool enable_random_segment_phase = true;
  std::size_t repeats = 1;
  std::optional<std::uint64_t> random_seed;
  PorosityKernelVariant porosity_variant = PorosityKernelVariant::baseline;
  FloatingPrecision floating_precision = FloatingPrecision::double_precision;
};

struct OutputConfig {
  std::optional<std::filesystem::path> vtk_file_name;
  std::optional<std::filesystem::path> morphology_file_name;
  std::vector<std::string> morphology_fields;
};

struct SimulationConfig {
  std::filesystem::path config_path;
  std::filesystem::path config_dir;
  std::vector<std::filesystem::path> scan_paths;
  SimulationParameters parameters;
  std::map<std::string, MeltPoolComponentConfig> melt_pool_data;
  bool has_rve = false;
  BoundBox rve{};
  OutputConfig output;
};

// Parse the repo's constrained YAML-style config format into a typed structure.
SimulationConfig parseConfig(const std::filesystem::path& config_path);

}  // namespace raptor
