#include "raptor/io.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace raptor {
namespace {

// Trim leading and trailing whitespace while keeping the parser simple and local.
std::string trim(const std::string& value) {
  const std::size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::vector<std::string> splitWhitespace(const std::string& line) {
  std::vector<std::string> tokens;
  std::istringstream stream(line);
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

// Compute Euclidean distance while keeping the scan parser single-pass.
template <typename Real>
Real pointDistance(const Vec3T<Real>& lhs, const Vec3T<Real>& rhs) {
  const Real dx = rhs[0] - lhs[0];
  const Real dy = rhs[1] - lhs[1];
  const Real dz = rhs[2] - lhs[2];
  return static_cast<Real>(std::sqrt(dx * dx + dy * dy + dz * dz));
}

}  // namespace

template <typename Real>
std::vector<std::vector<Real>> readData(const std::filesystem::path& file_name) {
  // Read either comma-delimited or whitespace-delimited numeric rows.
  std::ifstream input(file_name);
  if (!input) {
    throw std::runtime_error("Melt pool measurement file not found: " + file_name.string());
  }

  std::vector<std::vector<Real>> rows;
  std::string line;
  while (std::getline(input, line)) {
    const std::string cleaned = trim(line);
    if (cleaned.empty() || cleaned[0] == '#') {
      continue;
    }

    std::vector<Real> row;
    if (cleaned.find(',') != std::string::npos) {
      std::stringstream stream(cleaned);
      std::string token;
      while (std::getline(stream, token, ',')) {
        const std::string value = trim(token);
        if (!value.empty()) {
          row.push_back(static_cast<Real>(std::stod(value)));
        }
      }
    } else {
      for (const std::string& token : splitWhitespace(cleaned)) {
        row.push_back(static_cast<Real>(std::stod(token)));
      }
    }

    if (!row.empty()) {
      rows.push_back(std::move(row));
    }
  }

  return rows;
}

template <typename Real>
std::vector<PathVectorT<Real>> readScanPath(const std::filesystem::path& file_name) {
  // Parse scan-path records in a single pass while preserving the Python timing semantics.
  std::ifstream input(file_name);
  if (!input) {
    throw std::runtime_error("Scan path file not found: " + file_name.string());
  }

  std::string line;
  if (!std::getline(input, line)) {
    return {};
  }

  std::vector<PathVectorT<Real>> path_vectors;
  bool have_previous = false;
  Vec3T<Real> previous_position{};
  Real previous_time = static_cast<Real>(0);

  while (std::getline(input, line)) {
    const std::string cleaned = trim(line);
    if (cleaned.empty() || cleaned[0] == '#') {
      continue;
    }

    const std::vector<std::string> tokens = splitWhitespace(cleaned);
    if (tokens.size() < 6) {
      continue;
    }

    const int mode = static_cast<int>(std::stod(tokens[0]));
    const Vec3T<Real> position = {static_cast<Real>(std::stod(tokens[1])),
                                  static_cast<Real>(std::stod(tokens[2])),
                                  static_cast<Real>(std::stod(tokens[3]))};
    const Real parameter = static_cast<Real>(std::stod(tokens[5]));

    if (!have_previous) {
      previous_position = position;
      previous_time = mode == 1 ? parameter : static_cast<Real>(0);
      have_previous = true;
      continue;
    }

    const Real dt =
        mode == 1
            ? parameter
            : (parameter > static_cast<Real>(1.0e-12)
                   ? pointDistance(previous_position, position) / parameter
                   : static_cast<Real>(0));
    const Real current_time = previous_time + dt;

    if (mode == 0) {
      path_vectors.emplace_back(previous_position, position, previous_time, current_time);
    }

    previous_position = position;
    previous_time = current_time;
  }

  return path_vectors;
}

template std::vector<std::vector<float>> readData<float>(const std::filesystem::path& file_name);
template std::vector<std::vector<double>> readData<double>(const std::filesystem::path& file_name);
template std::vector<PathVectorT<float>> readScanPath<float>(const std::filesystem::path& file_name);
template std::vector<PathVectorT<double>> readScanPath<double>(const std::filesystem::path& file_name);

}  // namespace raptor
