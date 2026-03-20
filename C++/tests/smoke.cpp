#include "raptor/api.hpp"

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// Decode a base64-encoded appended payload from the VTI writer.
std::vector<std::uint8_t> decodeBase64(const std::string& encoded) {
  auto decodeChar = [](char ch) -> int {
    if (ch >= 'A' && ch <= 'Z') {
      return ch - 'A';
    }
    if (ch >= 'a' && ch <= 'z') {
      return ch - 'a' + 26;
    }
    if (ch >= '0' && ch <= '9') {
      return ch - '0' + 52;
    }
    if (ch == '+') {
      return 62;
    }
    if (ch == '/') {
      return 63;
    }
    if (ch == '=') {
      return -2;
    }
    return -1;
  };

  std::string filtered;
  filtered.reserve(encoded.size());
  for (char ch : encoded) {
    if (!std::isspace(static_cast<unsigned char>(ch))) {
      filtered.push_back(ch);
    }
  }

  require(filtered.size() % 4 == 0, "Base64 payload length must be a multiple of 4.");

  std::vector<std::uint8_t> decoded;
  decoded.reserve((filtered.size() / 4) * 3);

  for (std::size_t index = 0; index < filtered.size(); index += 4) {
    const int c0 = decodeChar(filtered[index]);
    const int c1 = decodeChar(filtered[index + 1]);
    const int c2 = decodeChar(filtered[index + 2]);
    const int c3 = decodeChar(filtered[index + 3]);
    require(c0 >= 0 && c1 >= 0, "Invalid base64 prefix.");
    require(c2 != -1 && c3 != -1, "Invalid base64 suffix.");

    const std::uint32_t triple = (static_cast<std::uint32_t>(c0) << 18) |
                                 (static_cast<std::uint32_t>(c1) << 12) |
                                 (static_cast<std::uint32_t>(c2 >= 0 ? c2 : 0) << 6) |
                                 static_cast<std::uint32_t>(c3 >= 0 ? c3 : 0);
    decoded.push_back(static_cast<std::uint8_t>((triple >> 16) & 0xFFU));
    if (c2 != -2) {
      decoded.push_back(static_cast<std::uint8_t>((triple >> 8) & 0xFFU));
    }
    if (c3 != -2) {
      decoded.push_back(static_cast<std::uint8_t>(triple & 0xFFU));
    }
  }

  return decoded;
}

void testSpectralModeBounds() {
  // Verify that the C++ port now rejects more requested modes than available samples.
  const std::vector<std::array<double, 2>> samples = {{0.0, 1.0}, {1.0, 2.0}, {2.0, 3.0}};
  bool threw = false;
  try {
    (void)raptor::computeSpectralComponents(samples, 4);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "computeSpectralComponents should reject n_modes > sample_count.");
}

void testMorphologyCentroidAndBBox() {
  // Verify that morphology centroids are computed from voxel centers and bbox is preserved.
  const std::array<std::size_t, 3> shape = {2, 1, 1};
  const std::vector<std::uint8_t> porosity = {1, 1};
  const raptor::MorphologyTable table =
      raptor::computeMorphology(porosity, shape, 1.0, {"centroid", "bbox"});

  require(table.rows.size() == 1, "Expected one connected pore component.");
  require(table.headers.size() == 9, "Expected centroid and bbox columns.");

  const std::vector<std::string>& row = table.rows[0];
  require(std::stod(row[0]) == 1.0, "Centroid x should be at the average voxel center.");
  require(std::stod(row[1]) == 0.5, "Centroid y should be centered in the single y voxel.");
  require(std::stod(row[2]) == 0.5, "Centroid z should be centered in the single z voxel.");
  require(std::stoul(row[3]) == 0 && std::stoul(row[4]) == 0 && std::stoul(row[5]) == 0,
          "Bounding box minima should match the component origin.");
  require(std::stoul(row[6]) == 2 && std::stoul(row[7]) == 1 && std::stoul(row[8]) == 1,
          "Bounding box maxima should match the exclusive upper bounds.");
}

void testVtiLayout() {
  // Verify that the writer preserves the VTK byte ordering expected by downstream readers.
  const std::filesystem::path output_path =
      std::filesystem::temp_directory_path() / "raptor_vti_layout_test.vti";
  const std::array<std::size_t, 3> shape = {2, 1, 2};
  const std::vector<std::uint8_t> porosity = {10, 11, 12, 13};
  raptor::writeVti({0.0, 0.0, 0.0}, 1.0, shape, porosity, output_path);

  std::ifstream input(output_path, std::ios::binary);
  const std::string file_bytes((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
  require(file_bytes.find("header_type=\"UInt32\"") != std::string::npos,
          "VTI header should use a 32-bit block size for compatibility.");
  require(file_bytes.find("<DataArray type=\"UInt8\" Name=\"porosity\" NumberOfComponents=\"1\" format=\"appended\" offset=\"0\"/>") != std::string::npos,
          "VTI DataArray metadata is missing or malformed.");

  const std::string marker = "<AppendedData encoding=\"base64\">_";
  const std::size_t payload_offset = file_bytes.find(marker);
  require(payload_offset != std::string::npos, "VTI appended-data marker was not found.");

  const std::size_t binary_offset = payload_offset + marker.size();
  const std::size_t payload_end = file_bytes.find("</AppendedData>", binary_offset);
  require(payload_end != std::string::npos, "VTI appended-data terminator was not found.");

  const std::vector<std::uint8_t> binary_bytes =
      decodeBase64(file_bytes.substr(binary_offset, payload_end - binary_offset));
  require(binary_bytes.size() >= sizeof(std::uint32_t) + 4,
          "VTI payload is shorter than expected.");

  std::uint32_t byte_count = 0;
  std::memcpy(&byte_count, binary_bytes.data(), sizeof(byte_count));
  require(byte_count == 4, "VTI payload byte count should match the porosity array size.");

  const unsigned char* payload = reinterpret_cast<const unsigned char*>(
      binary_bytes.data() + sizeof(byte_count));
  require(payload[0] == 10 && payload[1] == 12 && payload[2] == 11 && payload[3] == 13,
          "VTI payload ordering does not match the expected z-y-x traversal.");

  std::filesystem::remove(output_path);
}

void testCoreWorkflow() {
  // Exercise the smallest viable end-to-end library path without external files.
  const raptor::BoundBox bound_box = {{{0.0, 0.0, 0.0}, {1.0e-3, 2.0e-4, 2.0e-4}}};
  raptor::Grid grid = raptor::createGrid<double>(
      1.0e-4, std::optional<raptor::BoundBox>(bound_box), nullptr);

  std::vector<raptor::PathVector> path_vectors = {
      raptor::PathVector({0.0, 0.0, 0.0}, {1.0e-3, 0.0, 0.0}, 0.0, 1.0e-3)};
  path_vectors[0].setCoordinateFrame();

  const raptor::MeltPoolComponentInput component = {{{1.0e-4, 0.0, 0.0}}, 1, 1.0, 2.0};
  const std::map<std::string, raptor::MeltPoolComponentInput> inputs = {
      {"width", component}, {"depth", component}, {"height", component}};
  const raptor::MeltPool melt_pool = raptor::createMeltPool(inputs, false);

  const std::vector<std::uint8_t> porosity =
      raptor::computePorosity(grid, path_vectors, melt_pool);
  require(porosity.size() == grid.n_voxels, "Porosity size mismatch in the core workflow test.");
}

void testRepeatedPorosityWorkflow() {
  // Verify that repeated runs can stay on device and optionally skip final host readback.
  const raptor::BoundBox bound_box = {{{0.0, 0.0, 0.0}, {1.0e-3, 2.0e-4, 2.0e-4}}};
  const raptor::Grid grid = raptor::createGrid<double>(
      1.0e-4, std::optional<raptor::BoundBox>(bound_box), nullptr);

  std::vector<raptor::PathVector> path_vectors = {
      raptor::PathVector({0.0, 0.0, 0.0}, {1.0e-3, 0.0, 0.0}, 0.0, 1.0e-3)};
  path_vectors[0].setCoordinateFrame();

  const raptor::MeltPoolComponentInput component = {{{1.0e-4, 0.0, 0.0}}, 1, 1.0, 2.0};
  const std::map<std::string, raptor::MeltPoolComponentInput> inputs = {
      {"width", component}, {"depth", component}, {"height", component}};
  const raptor::MeltPool melt_pool = raptor::createMeltPool(inputs, false);

  const raptor::PorosityRunSummary no_copy_summary =
      raptor::computePorosityRuns(grid, path_vectors, melt_pool, 3, 42, false);
  require(no_copy_summary.melted_voxel_counts.size() == 3,
          "Repeat workflow should report one melted-count analysis per run.");
  require(no_copy_summary.final_porosity.empty(),
          "Repeat workflow should skip host porosity readback when no output needs it.");

  const raptor::PorosityRunSummary morphology_summary =
      raptor::computePorosityRuns(grid, path_vectors, melt_pool, 2, 42, false, {"area"});
  require(!morphology_summary.accumulated_morphology.headers.empty(),
          "Repeat workflow should accumulate morphology output without a final VTI readback.");
  require(morphology_summary.accumulated_morphology.headers.front() == "repeat",
          "Accumulated morphology should prepend a repeat column when multiple runs are combined.");
  require(morphology_summary.final_porosity.empty(),
          "Morphology accumulation alone should not force final porosity readback.");

  const raptor::PorosityRunSummary copy_summary =
      raptor::computePorosityRuns(grid, path_vectors, melt_pool, 2, 42, true);
  require(copy_summary.melted_voxel_counts.size() == 2,
          "Repeat workflow should preserve the requested repeat count.");
  require(copy_summary.final_porosity.size() == grid.n_voxels,
          "Repeat workflow should materialize the final porosity field when requested.");
}

void testPorosityVariantsAgree() {
  // Verify that all runtime-selectable porosity variants preserve the baseline results.
  const raptor::BoundBox bound_box = {{{0.0, 0.0, 0.0}, {1.0e-3, 2.0e-4, 2.0e-4}}};
  const raptor::Grid grid = raptor::createGrid<double>(
      1.0e-4, std::optional<raptor::BoundBox>(bound_box), nullptr);

  std::vector<raptor::PathVector> path_vectors = {
      raptor::PathVector({0.0, 0.0, 0.0}, {1.0e-3, 0.0, 0.0}, 0.0, 1.0e-3)};
  path_vectors[0].setCoordinateFrame();

  const raptor::MeltPoolComponentInput component = {{{1.0e-4, 0.0, 0.0}}, 1, 1.0, 2.0};
  const std::map<std::string, raptor::MeltPoolComponentInput> inputs = {
      {"width", component}, {"depth", component}, {"height", component}};
  const raptor::MeltPool melt_pool = raptor::createMeltPool(inputs, false);

  const raptor::PorosityRunSummary baseline = raptor::computePorosityRuns(
      grid, path_vectors, melt_pool, 2, 42, true, {"area"},
      raptor::PorosityKernelVariant::baseline);
  const std::vector<raptor::PorosityKernelVariant> variants = {
      raptor::PorosityKernelVariant::cached,
      raptor::PorosityKernelVariant::seed_batch4,
      raptor::PorosityKernelVariant::team_tile_seed_batch4,
      raptor::PorosityKernelVariant::auto_select};

  for (const raptor::PorosityKernelVariant variant : variants) {
    const raptor::PorosityRunSummary summary =
        raptor::computePorosityRuns(grid, path_vectors, melt_pool, 2, 42, true, {"area"},
                                    variant);
    require(summary.melted_voxel_counts == baseline.melted_voxel_counts,
            "Porosity variant changed the melted voxel counts.");
    require(summary.final_porosity == baseline.final_porosity,
            "Porosity variant changed the final porosity field.");
    require(summary.accumulated_morphology.rows.size() == baseline.accumulated_morphology.rows.size(),
            "Porosity variant changed the accumulated morphology row count.");
  }
}

void testSinglePrecisionWorkflow() {
  // Verify that the templated float workflow executes end-to-end and returns valid output.
  const raptor::BoundBoxT<float> bound_box = {{{0.0f, 0.0f, 0.0f}, {1.0e-3f, 2.0e-4f, 2.0e-4f}}};
  const raptor::GridT<float> grid =
      raptor::createGrid<float>(1.0e-4f, std::optional<raptor::BoundBoxT<float>>(bound_box),
                                nullptr);

  std::vector<raptor::PathVectorT<float>> path_vectors = {
      raptor::PathVectorT<float>({0.0f, 0.0f, 0.0f}, {1.0e-3f, 0.0f, 0.0f}, 0.0f, 1.0e-3f)};
  path_vectors[0].setCoordinateFrame();

  const raptor::MeltPoolComponentInputT<float> component = {{{1.0e-4f, 0.0f, 0.0f}}, 1, 1.0f, 2.0f};
  const std::map<std::string, raptor::MeltPoolComponentInputT<float>> inputs = {
      {"width", component}, {"depth", component}, {"height", component}};
  const raptor::MeltPoolT<float> melt_pool = raptor::createMeltPool<float>(inputs, false);

  const raptor::PorosityRunSummary summary =
      raptor::computePorosityRuns<float>(grid, path_vectors, melt_pool, 1, 7, true, {"area"});
  require(summary.melted_voxel_counts.size() == 1,
          "Single-precision workflow should report one repeat result.");
  require(summary.final_porosity.size() == grid.n_voxels,
          "Single-precision workflow should materialize the final porosity field.");
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  try {
    testSpectralModeBounds();
    testMorphologyCentroidAndBBox();
    testVtiLayout();
    testCoreWorkflow();
    testRepeatedPorosityWorkflow();
    testPorosityVariantsAgree();
    testSinglePrecisionWorkflow();
  } catch (const std::exception& error) {
    std::cerr << "Smoke test failed: " << error.what() << '\n';
    Kokkos::finalize();
    return EXIT_FAILURE;
  }

  Kokkos::finalize();
  return EXIT_SUCCESS;
}
