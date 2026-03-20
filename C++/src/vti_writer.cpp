#include "raptor/api.hpp"

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace raptor {
namespace {

// Encode a binary payload using the base64 alphabet expected by VTK XML writers.
std::string encodeBase64(const std::vector<std::uint8_t>& input) {
  static constexpr char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string output;
  output.reserve(((input.size() + 2) / 3) * 4);

  std::size_t index = 0;
  while (index < input.size()) {
    const std::size_t remaining = input.size() - index;
    const std::uint32_t byte0 = input[index++];
    const std::uint32_t byte1 = remaining > 1 ? input[index++] : 0U;
    const std::uint32_t byte2 = remaining > 2 ? input[index++] : 0U;

    const std::uint32_t triple = (byte0 << 16) | (byte1 << 8) | byte2;
    output.push_back(kAlphabet[(triple >> 18) & 0x3FU]);
    output.push_back(kAlphabet[(triple >> 12) & 0x3FU]);
    output.push_back(remaining > 1 ? kAlphabet[(triple >> 6) & 0x3FU] : '=');
    output.push_back(remaining > 2 ? kAlphabet[triple & 0x3FU] : '=');
  }

  return output;
}

// Append a 32-bit little-endian length header in the same block layout used by VTK XML.
void appendUInt32LittleEndian(std::vector<std::uint8_t>& buffer, std::uint32_t value) {
  buffer.push_back(static_cast<std::uint8_t>(value & 0xFFU));
  buffer.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFFU));
  buffer.push_back(static_cast<std::uint8_t>((value >> 16) & 0xFFU));
  buffer.push_back(static_cast<std::uint8_t>((value >> 24) & 0xFFU));
}

}  // namespace

template <typename Real>
void writeVti(const Vec3T<Real>& origin, Real voxel_resolution,
              const std::array<std::size_t, 3>& shape,
              const std::vector<std::uint8_t>& porosity,
              const std::filesystem::path& output_path) {
  // Repack the porosity field into VTK's point-data ordering and emit raw appended XML.
  const std::size_t nx = shape[0];
  const std::size_t ny = shape[1];
  const std::size_t nz = shape[2];
  if (porosity.size() != nx * ny * nz) {
    throw std::invalid_argument("Porosity array size does not match the grid shape.");
  }

  std::vector<std::uint8_t> vtk_order;
  vtk_order.reserve(porosity.size());
  for (std::size_t z = 0; z < nz; ++z) {
    for (std::size_t y = 0; y < ny; ++y) {
      for (std::size_t x = 0; x < nx; ++x) {
        const std::size_t index = ((x * ny) + y) * nz + z;
        vtk_order.push_back(porosity[index]);
      }
    }
  }

  std::ofstream output(output_path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Failed to open VTI output file: " + output_path.string());
  }

  if (vtk_order.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("VTI payload is too large for a 32-bit XML header.");
  }

  std::vector<std::uint8_t> appended_block;
  appended_block.reserve(sizeof(std::uint32_t) + vtk_order.size());
  appendUInt32LittleEndian(appended_block, static_cast<std::uint32_t>(vtk_order.size()));
  appended_block.insert(appended_block.end(), vtk_order.begin(), vtk_order.end());

  output << "<?xml version=\"1.0\"?>\n";
  output << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
  output << "  <ImageData WholeExtent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 "
         << (nz - 1) << "\" Origin=\"" << origin[0] << ' ' << origin[1] << ' ' << origin[2]
         << "\" Spacing=\"" << voxel_resolution << ' ' << voxel_resolution << ' '
         << voxel_resolution << "\">\n";
  output << "    <Piece Extent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1)
         << "\">\n";
  output << "      <PointData Scalars=\"porosity\">\n";
  output << "        <DataArray type=\"UInt8\" Name=\"porosity\" NumberOfComponents=\"1\" format=\"appended\" offset=\"0\"/>\n";
  output << "      </PointData>\n";
  output << "      <CellData/>\n";
  output << "    </Piece>\n";
  output << "  </ImageData>\n";
  output << "  <AppendedData encoding=\"base64\">_";
  output << encodeBase64(appended_block);
  output << "\n  </AppendedData>\n";
  output << "</VTKFile>\n";
}

template void writeVti<float>(const Vec3T<float>&, float, const std::array<std::size_t, 3>&,
                              const std::vector<std::uint8_t>&,
                              const std::filesystem::path&);
template void writeVti<double>(const Vec3T<double>&, double, const std::array<std::size_t, 3>&,
                               const std::vector<std::uint8_t>&,
                               const std::filesystem::path&);

}  // namespace raptor
