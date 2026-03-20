#include "raptor/config.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace raptor {
namespace {

// Keep the config parser intentionally narrow to the documented repo schema.
std::string trim(const std::string& value) {
  const std::size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::string stripQuotes(const std::string& value) {
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

std::string stripComment(const std::string& line) {
  const std::size_t comment = line.find('#');
  return comment == std::string::npos ? line : line.substr(0, comment);
}

int indentation(const std::string& line) {
  int indent = 0;
  for (char ch : line) {
    if (ch == ' ') {
      ++indent;
    } else {
      break;
    }
  }
  return indent;
}

std::pair<std::string, std::string> splitKeyValue(const std::string& line) {
  const std::size_t delimiter = line.find(':');
  if (delimiter == std::string::npos) {
    return {trim(line), ""};
  }
  return {trim(line.substr(0, delimiter)), trim(line.substr(delimiter + 1))};
}

bool parseBool(const std::string& value) {
  const std::string lowered = value == "True" ? "true" : value == "False" ? "false" : value;
  if (lowered == "true") {
    return true;
  }
  if (lowered == "false") {
    return false;
  }
  throw std::invalid_argument("Invalid boolean value: " + value);
}

double parseDouble(const std::string& value) {
  return std::stod(stripQuotes(value));
}

Vec3 parseArray3(const std::string& value) {
  const std::string cleaned = trim(value.substr(1, value.size() - 2));
  std::stringstream stream(cleaned);
  std::string token;
  Vec3 result{};
  for (int index = 0; index < 3; ++index) {
    if (!std::getline(stream, token, ',')) {
      throw std::invalid_argument("Expected a three-value array: " + value);
    }
    result[index] = std::stod(trim(token));
  }
  return result;
}

}  // namespace

SimulationConfig parseConfig(const std::filesystem::path& config_path) {
  // Parse the small YAML-like config used by the example and CLI workflow.
  std::ifstream input(config_path);
  if (!input) {
    throw std::runtime_error("Config file not found: " + config_path.string());
  }

  SimulationConfig config;
  config.config_path = std::filesystem::absolute(config_path);
  config.config_dir = config.config_path.parent_path();

  std::string top_section;
  std::string nested_section;
  std::string melt_component;
  std::string line;
  while (std::getline(input, line)) {
    const std::string uncommented = stripComment(line);
    const std::string cleaned = trim(uncommented);
    if (cleaned.empty()) {
      continue;
    }

    const int indent = indentation(uncommented);
    if (indent == 0 && cleaned.back() == ':') {
      top_section = cleaned.substr(0, cleaned.size() - 1);
      nested_section.clear();
      melt_component.clear();
      continue;
    }

    if (top_section == "scan_paths" && cleaned.rfind("- ", 0) == 0) {
      config.scan_paths.emplace_back(stripQuotes(trim(cleaned.substr(2))));
      continue;
    }

    if (top_section == "parameters") {
      const auto [key, value] = splitKeyValue(cleaned);
      if (key == "layer_height") {
        config.parameters.layer_height = parseDouble(value);
      } else if (key == "voxel_resolution") {
        config.parameters.voxel_resolution = parseDouble(value);
      } else if (key == "enable_random_segment_phase") {
        config.parameters.enable_random_segment_phase = parseBool(stripQuotes(value));
      } else if (key == "repeats" || key == "number_of_repeats") {
        config.parameters.repeats = static_cast<std::size_t>(std::llround(parseDouble(value)));
      } else if (key == "random_seed") {
        config.parameters.random_seed =
            static_cast<std::uint64_t>(std::llround(parseDouble(value)));
      } else if (key == "porosity_variant") {
        config.parameters.porosity_variant = parsePorosityKernelVariant(stripQuotes(value));
      } else if (key == "floating_precision" || key == "real_type" || key == "precision") {
        config.parameters.floating_precision = parseFloatingPrecision(stripQuotes(value));
      }
      continue;
    }

    if (top_section == "melt_pool_data") {
      if (indent == 2 && cleaned.back() == ':') {
        melt_component = cleaned.substr(0, cleaned.size() - 1);
        config.melt_pool_data[melt_component] = MeltPoolComponentConfig{};
        if (melt_component == "width") {
          config.melt_pool_data[melt_component].shape = 2.0;
        }
        continue;
      }

      const auto [key, value] = splitKeyValue(cleaned);
      MeltPoolComponentConfig& component = config.melt_pool_data[melt_component];
      if (key == "type") {
        component.type = stripQuotes(value);
      } else if (key == "file_name") {
        component.file_name = stripQuotes(value);
      } else if (key == "nmodes") {
        component.n_modes = static_cast<std::size_t>(std::llround(parseDouble(value)));
      } else if (key == "scale") {
        component.scale = parseDouble(value);
      } else if (key == "shape") {
        component.shape = parseDouble(value);
      }
      continue;
    }

    if (top_section == "rve") {
      const auto [key, value] = splitKeyValue(cleaned);
      if (key == "min_point") {
        config.rve[0] = parseArray3(value);
        config.has_rve = true;
      } else if (key == "max_point") {
        config.rve[1] = parseArray3(value);
        config.has_rve = true;
      }
      continue;
    }

    if (top_section == "output") {
      if (indent == 2 && cleaned.back() == ':') {
        nested_section = cleaned.substr(0, cleaned.size() - 1);
        continue;
      }
      if (nested_section == "vtk") {
        const auto [key, value] = splitKeyValue(cleaned);
        if (key == "file_name") {
          config.output.vtk_file_name = std::filesystem::path(stripQuotes(value));
        }
        continue;
      }
      if (nested_section == "morphology") {
        if (cleaned.rfind("- ", 0) == 0) {
          config.output.morphology_fields.push_back(stripQuotes(trim(cleaned.substr(2))));
          continue;
        }
        const auto [key, value] = splitKeyValue(cleaned);
        if (key == "file_name") {
          config.output.morphology_file_name = std::filesystem::path(stripQuotes(value));
        }
      }
    }
  }

  return config;
}

}  // namespace raptor
