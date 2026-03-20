#pragma once

#include "raptor/types.hpp"

namespace raptor {

// Read numeric time-series or spectral data from a text or CSV file.
template <typename Real = double>
std::vector<std::vector<Real>> readData(const std::filesystem::path& file_name);

// Read a scan-path file into the exposure segments used by the solver.
template <typename Real = double>
std::vector<PathVectorT<Real>> readScanPath(const std::filesystem::path& file_name);

}  // namespace raptor
