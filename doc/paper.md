---
title: 'Raptor: A Python Library for Porosity Predictions in Additive Manufacturing'
tags:
  - Python
  - Additive Manufacturing
authors:
  - name: Vamsi Subraveti
    orcid: 0000-0002-9205-8212
    affiliation: 1
  - name: John Coleman
    orcid: 0000-0002-7261-3143
    affiliation: 2
  - name: Alex Plotkowski
    orcid: 0000-0001-5471-8681
    affiliation: 2
  - name: Çağlar Oskay
    orcid:
    affiliation: 1    
affiliations:
 - name: Vanderbilt University, Nashville, TN, USA
   index: 1
 - name: Oak Ridge National Laboratory, Oak Ridge, TN, USA
   index: 2
date: 06 June 2025
bibliography: paper.bib
header-includes:
  - \usepackage{amsmath}
---

# Summary

Raptor is an efficient Python library for simulating stochastic lack-of-fusion (sLoF) defects in additive manufacturing (AM) processes. These defects arise from stochastic variations in the melt pool boundary leading to undermelting from insufficient overlap between adjacent melt pools or successive layers [@khairallah_physics_2016;@grasso_insitu_2017]. Performance variability of AM parts is a pressing challenge in qualification and certification of AM parts; this is in part due to the poorly understood formation rate and statistics of sLoF defects. Raptor is designed to capture the explicit morphologies of sLoF defects and their statistics to accelerate qualification and certification efforts of AM parts in critical applications.

# Statement of need

Metal AM processes show great promise in advancing manufacturing capabilities across a variety of industries. However, the quantification of uncertainty in the desired properties of AM parts is an ongoing challenge which spans many disciplines of engineering. A key driver in this challenge is the modeling of explicit sLoF defect geometries and their occurrence rates [@reddy_fatigue_2024;@berez_fatiguevariation_2022]. Experimental observations of sLoF show that these defects persist well into the previously determined optimal processing regime; their occurrence, however, is sparse [@miner_lof_2024]. The sparsity of the sLoF defects coupled with their impact on desirable properties necessitates a focused, collaborative modeling effort for the simulation, prediction, and mitigation of these defects to accelerate metal AM adoption throughout industry. 

# State of the field

The current modeling landscape for sLoF prediction is fairly sparse; initial work provided simple 2-dimensional estimations of defect volume fraction in the deterministic melt pool case [@tang_lof_2017]. This was extended to 3-dimensional geometry prediction, continuing to utilize the deterministic melt pool assumption [@subraveti_lof_2024]. More recently, uncertainty in the melt pool dimensions to augment sLoF predictions in a Tang-type semi-analytical geometric model has been introduced [@richter_analyticallof_2025]. The current state-of-the-art sLoF model builds on this approach with a Fourier-based expansion of the melt pool dimensions to account for temporal variability [@subraveti_sma_2025;@subraveti_process_2025]. This model solves for explicit melt pool overlaps, yielding sLoF geometries resultant from melt pool fluctuations. 

# Software design

In this manuscript, we introduce a significant improvement to state-of-the-art in the prediction and characterization of explicit sLoF geometries and the distribution of their morphological metrics. The novel parallelized algorithm in Raptor results in a multiple order-of-magnitude efficiency gain over the previous state-of-the-art serially timestepped approach. In Raptor, the melt pool dimensions are treated as a stochastic process informed by user-supplied data. These stochastic processes are efficiently sampled via a truncated cosine expansion of the full Fourier basis computed from the user-supplied data. The locally computed dimensions from the cosine expansion are used to define a Lamé curve, which is then used to mask a voxelized grid representing the printed volume of interest.

The voxel-masking approach has been used in the previously discussed models to assess the occurrence and morphology of unmelted regions corresponding to defects in a printed part. Our implementation of voxel-masking leverages a point-parallel approach, where the independence of each voxel in the domain relative to any other voxel allows for parallel processing of the masking operation. The core performance gain stems from the a priori computation of axis-aligned bounding boxes and oriented bounding boxes (AABBs and OBBs respectively) for each voxel-scan vector pair in the user-defined domain. The process is then simulated in parallel over each voxel, considering only the local melt pools which coincide with the plane of the voxel relative to the bounding box. If the voxel is inside a local melt pool boundary, it is marked as melted. This paradigm makes probabilistic analyses of explicit sLoF defect structures tractable. Raptor is able to process representative volume elements (RVEs) of edge size 0.5 mm with a resolution of 2.5 µm (8,000,000 voxels) within 1-4 seconds on a local workstation; this RVE simulation can be repeatedly queried to construct an ensemble of 1000 sLoF realizations in under an hour, or an equivalent scanned volume of 125 mm$^3$. The scale of this RVE ensemble is able to capture rare sLoF defect events and is targeted toward simulating defects responsible for deleterious part performance within the optimized printing regime [@reddy_fatigue_2024].

# Software features

The main software feature of Raptor is the generation of stochastic lack-of-fusion structures through the efficient propagation of input melt pool uncertainty. Raptor can handle either(a) melt pool dimension measurements over time, or (b) an array containing amplitude and frequency information. The user may input standard laser powder bed fusion parameters either through a .yaml file or by using the Raptor API endpoints. Raptor outputs an binary .vti file along with optional basic morphological metrics computed directly through scikit-image. Additional features are available through the usage of the Raptor API. Users can customize the melt pool shape with the Lamé shape coefficient and manually tweak individual melt pool parameters, such as aspect ratio and size. Utilities include a scan path generator class, which allows for the quick construction of scan vectors corresponding to common laser powder bed fusion settings; these can all be adjusted based on user preference for a wide variety of processing conditions. Example cases with both the .yaml and scripted API examples have been provided with detailed comments.

Applications of Raptor to the AM modeling community are numerous, stemming from the potential for rapid probabilistic assessments of defects coupled with their explicit morphologies. Two primary directions of applications are posited: the forward and inverse applications. The forward application would encompass problems such as defect structure prediction at some user-defined process parameters. The user would perform a high-fidelity single/multitrack simulation or characterize a single/multitrack experiment to input to Raptor. Then, the resulting sLoF defect structures and the distributions of the relevant quantities of interest (QoIs) can be constructed. For example, a user may want to simulate sLoF formation at the given process parameters for a sub-region of a build that was determined to undergo extreme loading conditions. The inverse application would include design problems subject to constraints on defect structure distributions and occurrence. As manufacturer-recommended parameters have been shown to still produce rare-event sLoF defects, the inverse problem is highly relevant to exploring the LPBF process space with a clearer view of sLoF defect statistics. An example of an inverse problem would be to minimize defect structure occurrence rate while maximizing build efficiency via spacing parameters.

# AI usage disclosure

This software was initially developed by V. Subraveti and J. Coleman; generative AI was used to refactor the codebase with readable variables and generate skeleton test cases for optimal coverage. Afterwards, the two lead developers revised this output to verify the structure, logic, and results from the software. 

# Acknowledgements

This manuscript has been authored by UT-Battelle, LLC under Contract No. DE-AC05-
00OR22725 with the U.S. Department of Energy (DOE). The publisher, by accepting the
article for publication, acknowledges that the United States Government retains a non-exclusive,
paid-up, irrevocable, world-wide license to publish or reproduce the published form of this
manuscript, or allow others to do so, for United States Government purposes. The DOE will
provide public access to these results of federally sponsored research in accordance with the
DOE Public Access Plan.
The development of Raptor was sponsored by the DOE Advanced Materials & Manufacturing Technologies Office and utilized resources at the Oak Ridge National
Laboratory Manufacturing Demonstration Facility.
This work was supported by a Space Technology Research Institutes grant from NASA’s Space Technology Research Grants Program under Grant #80NSSC23K1342.

# References
