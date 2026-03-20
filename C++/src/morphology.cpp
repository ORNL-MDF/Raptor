#include "raptor/api.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace raptor {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Convert numeric values to stable CSV strings without dropping precision gratuitously.
std::string toString(double value) {
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

std::string toString(std::size_t value) {
  return std::to_string(value);
}

// Flatten a 3D ijk coordinate into the x-major storage order used throughout the C++ port.
KOKKOS_INLINE_FUNCTION int flatIndex(int x, int y, int z, int ny, int nz) {
  return ((x * ny) + y) * nz + z;
}

// Check whether the requested morphology property is implemented by the device workflow.
bool isSupportedField(const std::string& field) {
  static const std::unordered_set<std::string> supported = {
      "label", "area", "centroid", "equivalent_diameter_area", "bbox",
      "area_bbox", "extent", "area_filled"};
  return supported.count(field) > 0;
}

// Build the CSV column layout that mirrors the Python regionprops table output.
MorphologyTable createMorphologyTableTemplate(
    const std::vector<std::string>& morphology_fields) {
  MorphologyTable table;
  for (const std::string& field : morphology_fields) {
    if (field == "centroid") {
      table.headers.push_back("centroid-0");
      table.headers.push_back("centroid-1");
      table.headers.push_back("centroid-2");
    } else if (field == "bbox") {
      for (int dim = 0; dim < 6; ++dim) {
        table.headers.push_back("bbox-" + std::to_string(dim));
      }
    } else {
      table.headers.push_back(field);
    }
  }
  return table;
}

}  // namespace

template <typename Real>
MorphologyTable computeMorphologyFromDevice(
    const Kokkos::View<std::uint8_t*>& porosity_view,
    const std::array<std::size_t, 3>& shape, Real voxel_resolution,
    const std::vector<std::string>& morphology_fields) {
  Kokkos::Profiling::ScopedRegion region("raptor::compute_morphology");

  // Validate the requested field set before allocating any device buffers.
  for (const std::string& field : morphology_fields) {
    if (!isSupportedField(field)) {
      throw std::invalid_argument("Unsupported morphology field: " + field);
    }
  }

  MorphologyTable table = createMorphologyTableTemplate(morphology_fields);
  const std::size_t n_voxels_size_t = shape[0] * shape[1] * shape[2];
  if (n_voxels_size_t == 0) {
    return table;
  }

  const int nx = static_cast<int>(shape[0]);
  const int ny = static_cast<int>(shape[1]);
  const int nz = static_cast<int>(shape[2]);
  const int n_voxels = static_cast<int>(n_voxels_size_t);

  // Initialize each pore voxel with its own label before iterative label propagation.
  Kokkos::View<int*> labels_view("labels", n_voxels);
  Kokkos::View<int*> next_labels_view("next_labels", n_voxels);
  {
    Kokkos::Profiling::ScopedRegion init_region("raptor::morphology::initialize_labels");
    Kokkos::parallel_for(
        "raptor::morphology::seed_labels", Kokkos::RangePolicy<>(0, n_voxels),
        KOKKOS_LAMBDA(const int voxel_index) {
          const int label = porosity_view(voxel_index) != 0 ? voxel_index + 1 : 0;
          labels_view(voxel_index) = label;
          next_labels_view(voxel_index) = label;
        });
  }

  // Propagate the minimum connected label through each pore component until the labeling settles.
  {
    Kokkos::Profiling::ScopedRegion label_region("raptor::morphology::label_components");
    Kokkos::View<int> changed_view("changed");
    int changed = 0;
    do {
      Kokkos::deep_copy(changed_view, 0);
      Kokkos::parallel_for(
          "raptor::morphology::relax_labels", Kokkos::RangePolicy<>(0, n_voxels),
          KOKKOS_LAMBDA(const int voxel_index) {
            const int current_label = labels_view(voxel_index);
            if (current_label == 0) {
              next_labels_view(voxel_index) = 0;
              return;
            }

            const int z_index = voxel_index % nz;
            const int yz_index = voxel_index / nz;
            const int y_index = yz_index % ny;
            const int x_index = yz_index / ny;

            int min_label = current_label;
            for (int dx = -1; dx <= 1; ++dx) {
              for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                  if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                  }
                  const int nx_index = x_index + dx;
                  const int ny_index = y_index + dy;
                  const int nz_index = z_index + dz;
                  if (nx_index < 0 || ny_index < 0 || nz_index < 0 || nx_index >= nx ||
                      ny_index >= ny || nz_index >= nz) {
                    continue;
                  }

                  const int neighbor_index = flatIndex(nx_index, ny_index, nz_index, ny, nz);
                  if (porosity_view(neighbor_index) == 0) {
                    continue;
                  }
                  const int neighbor_label = labels_view(neighbor_index);
                  if (neighbor_label != 0 && neighbor_label < min_label) {
                    min_label = neighbor_label;
                  }
                }
              }
            }

            next_labels_view(voxel_index) = min_label;
            if (min_label != current_label) {
              Kokkos::atomic_increment(&changed_view());
            }
          });
      Kokkos::deep_copy(changed, changed_view);
      std::swap(labels_view, next_labels_view);
    } while (changed > 0);
  }

  // Count sparse root labels and compact the surviving components into a dense 1..N id space.
  Kokkos::View<int*> root_counts_view("root_counts", n_voxels + 1);
  Kokkos::View<int*> compact_ids_view("compact_ids", n_voxels + 1);
  int n_components = 0;
  {
    Kokkos::Profiling::ScopedRegion count_region("raptor::morphology::count_components");
    Kokkos::deep_copy(root_counts_view, 0);
    Kokkos::parallel_for(
        "raptor::morphology::count_roots", Kokkos::RangePolicy<>(0, n_voxels),
        KOKKOS_LAMBDA(const int voxel_index) {
          const int label = labels_view(voxel_index);
          if (label != 0) {
            Kokkos::atomic_increment(&root_counts_view(label));
          }
        });

    Kokkos::parallel_scan(
        "raptor::morphology::compact_roots", Kokkos::RangePolicy<>(0, n_voxels + 1),
        KOKKOS_LAMBDA(const int root_index, int& update, const bool final_pass) {
          const bool keep_component = root_counts_view(root_index) >= 2;
          if (final_pass) {
            compact_ids_view(root_index) = keep_component ? update + 1 : 0;
          }
          if (keep_component) {
            ++update;
          }
        },
        n_components);
  }

  if (n_components == 0) {
    return table;
  }

  // Reduce component counts, bounding boxes, and centroids directly on the device.
  Kokkos::View<int*> component_counts_view("component_counts", n_components + 1);
  Kokkos::View<int**, Kokkos::LayoutRight> bbox_view("bbox", n_components + 1, 6);
  Kokkos::View<Real**, Kokkos::LayoutRight> centroid_sums_view("centroid_sums",
                                                               n_components + 1, 3);
  {
    Kokkos::Profiling::ScopedRegion accumulate_region("raptor::morphology::accumulate_properties");
    Kokkos::deep_copy(component_counts_view, 0);
    Kokkos::deep_copy(centroid_sums_view, 0.0);
    Kokkos::parallel_for(
        "raptor::morphology::initialize_bbox", Kokkos::RangePolicy<>(0, n_components + 1),
        KOKKOS_LAMBDA(const int component) {
          bbox_view(component, 0) = nx;
          bbox_view(component, 1) = ny;
          bbox_view(component, 2) = nz;
          bbox_view(component, 3) = -1;
          bbox_view(component, 4) = -1;
          bbox_view(component, 5) = -1;
        });

    Kokkos::parallel_for(
        "raptor::morphology::reduce_components", Kokkos::RangePolicy<>(0, n_voxels),
        KOKKOS_LAMBDA(const int voxel_index) {
          const int label = labels_view(voxel_index);
          if (label == 0) {
            return;
          }
          const int component = compact_ids_view(label);
          if (component == 0) {
            return;
          }

          const int z_index = voxel_index % nz;
          const int yz_index = voxel_index / nz;
          const int y_index = yz_index % ny;
          const int x_index = yz_index / ny;

          Kokkos::atomic_increment(&component_counts_view(component));
          Kokkos::atomic_min(&bbox_view(component, 0), x_index);
          Kokkos::atomic_min(&bbox_view(component, 1), y_index);
          Kokkos::atomic_min(&bbox_view(component, 2), z_index);
          Kokkos::atomic_max(&bbox_view(component, 3), x_index);
          Kokkos::atomic_max(&bbox_view(component, 4), y_index);
          Kokkos::atomic_max(&bbox_view(component, 5), z_index);
          Kokkos::atomic_add(&centroid_sums_view(component, 0),
                             (static_cast<Real>(x_index) + static_cast<Real>(0.5)) *
                                 voxel_resolution);
          Kokkos::atomic_add(&centroid_sums_view(component, 1),
                             (static_cast<Real>(y_index) + static_cast<Real>(0.5)) *
                                 voxel_resolution);
          Kokkos::atomic_add(&centroid_sums_view(component, 2),
                             (static_cast<Real>(z_index) + static_cast<Real>(0.5)) *
                                 voxel_resolution);
        });
  }

  // Read back the compact per-component summaries and format the requested table rows on host.
  Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace> component_counts_host(
      "component_counts_host", n_components + 1);
  Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace> bbox_host("bbox_host",
                                                                        n_components + 1, 6);
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace> centroid_sums_host(
      "centroid_sums_host", n_components + 1, 3);
  {
    Kokkos::Profiling::ScopedRegion readback_region("raptor::morphology::readback");
    Kokkos::deep_copy(component_counts_host, component_counts_view);
    Kokkos::deep_copy(bbox_host, bbox_view);
    Kokkos::deep_copy(centroid_sums_host, centroid_sums_view);
  }

  const Real voxel_volume = voxel_resolution * voxel_resolution * voxel_resolution;
  table.rows.reserve(static_cast<std::size_t>(n_components));
  for (int component = 1; component <= n_components; ++component) {
    const std::size_t component_size = static_cast<std::size_t>(component_counts_host(component));
    if (component_size < 2) {
      continue;
    }

    const int min_x = bbox_host(component, 0);
    const int min_y = bbox_host(component, 1);
    const int min_z = bbox_host(component, 2);
    const int max_x = bbox_host(component, 3);
    const int max_y = bbox_host(component, 4);
    const int max_z = bbox_host(component, 5);
    const Real area = static_cast<Real>(component_size) * voxel_volume;
    const Real area_bbox =
        static_cast<Real>((max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1)) *
        voxel_volume;
    const Real extent = area_bbox > static_cast<Real>(0) ? area / area_bbox : static_cast<Real>(0);
    const Real equivalent_diameter =
        static_cast<Real>(std::cbrt((6.0 * static_cast<double>(area)) / kPi));
    const Real centroid_x =
        centroid_sums_host(component, 0) / static_cast<Real>(component_size);
    const Real centroid_y =
        centroid_sums_host(component, 1) / static_cast<Real>(component_size);
    const Real centroid_z =
        centroid_sums_host(component, 2) / static_cast<Real>(component_size);

    std::vector<std::string> row;
    for (const std::string& field : morphology_fields) {
      if (field == "label") {
        row.push_back(toString(static_cast<std::size_t>(component)));
      } else if (field == "area") {
        row.push_back(toString(area));
      } else if (field == "area_filled") {
        row.push_back(toString(area));
      } else if (field == "area_bbox") {
        row.push_back(toString(area_bbox));
      } else if (field == "extent") {
        row.push_back(toString(extent));
      } else if (field == "equivalent_diameter_area") {
        row.push_back(toString(equivalent_diameter));
      } else if (field == "centroid") {
        row.push_back(toString(centroid_x));
        row.push_back(toString(centroid_y));
        row.push_back(toString(centroid_z));
      } else if (field == "bbox") {
        row.push_back(toString(static_cast<std::size_t>(min_x)));
        row.push_back(toString(static_cast<std::size_t>(min_y)));
        row.push_back(toString(static_cast<std::size_t>(min_z)));
        row.push_back(toString(static_cast<std::size_t>(max_x + 1)));
        row.push_back(toString(static_cast<std::size_t>(max_y + 1)));
        row.push_back(toString(static_cast<std::size_t>(max_z + 1)));
      }
    }
    table.rows.push_back(row);
  }

  return table;
}

template <typename Real>
MorphologyTable computeMorphology(const std::vector<std::uint8_t>& porosity,
                                  const std::array<std::size_t, 3>& shape,
                                  Real voxel_resolution,
                                  const std::vector<std::string>& morphology_fields) {
  const std::size_t n_voxels = shape[0] * shape[1] * shape[2];
  if (porosity.size() != n_voxels) {
    throw std::invalid_argument("Porosity array size does not match the supplied shape.");
  }

  // Copy the host porosity field to device, then reuse the device-native morphology workflow.
  Kokkos::View<std::uint8_t*> porosity_view("porosity", static_cast<int>(n_voxels));
  {
    Kokkos::Profiling::ScopedRegion copy_region("raptor::morphology::copy_input");
    Kokkos::View<const std::uint8_t*, Kokkos::LayoutRight, Kokkos::HostSpace> porosity_host(
        porosity.data(), static_cast<int>(n_voxels));
    Kokkos::deep_copy(porosity_view, porosity_host);
  }
  return computeMorphologyFromDevice<Real>(porosity_view, shape, voxel_resolution,
                                           morphology_fields);
}

void writeMorphology(const MorphologyTable& table,
                     const std::filesystem::path& output_path) {
  // Write a flat CSV that mirrors the regionprops_table style column layout.
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Failed to open morphology output file: " + output_path.string());
  }

  for (std::size_t column = 0; column < table.headers.size(); ++column) {
    if (column > 0) {
      output << ',';
    }
    output << table.headers[column];
  }
  output << '\n';

  for (const std::vector<std::string>& row : table.rows) {
    for (std::size_t column = 0; column < row.size(); ++column) {
      if (column > 0) {
        output << ',';
      }
      output << row[column];
    }
    output << '\n';
  }
}

template MorphologyTable computeMorphologyFromDevice<float>(
    const Kokkos::View<std::uint8_t*>&, const std::array<std::size_t, 3>&, float,
    const std::vector<std::string>&);
template MorphologyTable computeMorphologyFromDevice<double>(
    const Kokkos::View<std::uint8_t*>&, const std::array<std::size_t, 3>&, double,
    const std::vector<std::string>&);
template MorphologyTable computeMorphology<float>(
    const std::vector<std::uint8_t>&, const std::array<std::size_t, 3>&, float,
    const std::vector<std::string>&);
template MorphologyTable computeMorphology<double>(
    const std::vector<std::uint8_t>&, const std::array<std::size_t, 3>&, double,
    const std::vector<std::string>&);

}  // namespace raptor
