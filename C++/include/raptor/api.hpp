#pragma once

#include <Kokkos_Core.hpp>

#include "raptor/config.hpp"
#include "raptor/io.hpp"
#include "raptor/types.hpp"

namespace raptor {

struct PorosityRunSummary {
  std::vector<std::size_t> melted_voxel_counts;
  std::vector<std::uint8_t> final_porosity;
  MorphologyTable accumulated_morphology;
  std::uint64_t base_seed = 0;
  PorosityKernelVariant variant_used = PorosityKernelVariant::baseline;
};

// Build a voxel grid from either an explicit RVE or an inferred path-vector extent.
template <typename Real = double>
GridT<Real> createGrid(Real voxel_resolution,
                       const std::optional<BoundBoxT<Real>>& bound_box,
                       const std::vector<PathVectorT<Real>>* path_vectors);

// Generate a layered hatch scan strategy from process parameters.
template <typename Real = double>
std::vector<PathVectorT<Real>> createPathVectors(const BoundBoxT<Real>& bound_box, Real power,
                                                 Real scan_speed, Real hatch_spacing,
                                                 Real layer_height, Real rotation_degrees,
                                                 Real scan_extension, int extra_layers);

// Compute the leading spectral modes used to reconstruct the melt-pool oscillations.
template <typename Real = double>
std::vector<ModeDataT<Real>> computeSpectralComponents(
    const std::vector<std::array<Real, 2>>& melt_pool_data, std::size_t n_modes);

// Convert width/depth/height inputs into the packed melt-pool state used by the solver.
template <typename Real = double>
MeltPoolT<Real> createMeltPool(
    const std::map<std::string, MeltPoolComponentInputT<Real>>& melt_pool_dict,
    bool enable_random_phases);

// Compute the melted mask over the voxel grid using the prepared path vectors and melt pool.
template <typename Real = double>
std::vector<std::uint8_t> computeMeltMask(const GridT<Real>& grid,
                                          const MeltPoolT<Real>& melt_pool,
                                          const std::vector<PathVectorT<Real>>& path_vectors);

// Compute the final porosity field with values matching the Python workflow.
template <typename Real = double>
std::vector<std::uint8_t> computePorosity(const GridT<Real>& grid,
                                          std::vector<PathVectorT<Real>>& path_vectors,
                                          const MeltPoolT<Real>& melt_pool);

// Run one or more porosity simulations while keeping static data resident on the device.
template <typename Real = double>
PorosityRunSummary computePorosityRuns(
    const GridT<Real>& grid, const std::vector<PathVectorT<Real>>& path_vectors,
    const MeltPoolT<Real>& melt_pool, std::size_t repeats, std::uint64_t base_seed,
    bool copy_final_porosity, const std::vector<std::string>& morphology_fields = {},
    PorosityKernelVariant variant = PorosityKernelVariant::auto_select);

// Write the porosity field as VTK ImageData without requiring the VTK C++ library.
template <typename Real = double>
void writeVti(const Vec3T<Real>& origin, Real voxel_resolution,
              const std::array<std::size_t, 3>& shape,
              const std::vector<std::uint8_t>& porosity,
              const std::filesystem::path& output_path);

// Compute supported morphology fields for connected pore regions.
template <typename Real = double>
MorphologyTable computeMorphology(const std::vector<std::uint8_t>& porosity,
                                  const std::array<std::size_t, 3>& shape,
                                  Real voxel_resolution,
                                  const std::vector<std::string>& morphology_fields);

// Compute supported morphology fields directly from a device-resident porosity view.
template <typename Real = double>
MorphologyTable computeMorphologyFromDevice(
    const Kokkos::View<std::uint8_t*>& porosity_view,
    const std::array<std::size_t, 3>& shape, Real voxel_resolution,
    const std::vector<std::string>& morphology_fields);

// Persist morphology rows to a CSV file.
void writeMorphology(const MorphologyTable& table,
                     const std::filesystem::path& output_path);

}  // namespace raptor
