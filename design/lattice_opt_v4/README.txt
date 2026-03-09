LATTICE WING OPTIMIZATION DEBUG WORKFLOW
========================================

Purpose
-------
This codebase is a debugging and development framework for graph-based structural
optimization of a lattice wing-box. The current goal is not yet a final optimizer,
but rather to establish a trustworthy workflow that can:

1. generate an interpretable 3D lattice wing structure,
2. apply structural loads and clamped root boundary conditions,
3. solve intact and damaged truss responses,
4. evaluate connectivity and basic graph metrics,
5. attempt topology changes through grouped pruning,
6. save enough debug data to understand why a design passes or fails.

At the current stage, this framework is best viewed as a research sandbox for
testing:
- structural graph generation,
- damage modeling,
- grouped topology modification,
- graph-theoretic metrics,
- and the interaction between graph connectivity and structural feasibility.

Code File Responsibilities
==========================

1. optimize_v4_connected.py
---------------------------
This is the main driver script.

Primary responsibilities:
- defines the overall problem setup,
- chooses which lattice generator to use,
- sets material properties and load level,
- initializes member areas,
- runs the intact structural solve,
- runs the damaged structural solve,
- updates member sizes,
- attempts grouped pruning,
- records iteration history,
- writes HDF5 debug output,
- calls plotting/visualization functions.

This file is where the overall optimization workflow lives.

Important variables in this file:
- lattice_type:
  chooses between square and octet graph families.
- E_modulus:
  Young's modulus of the truss material.
- density:
  material density used only in mass calculation.
- sigma_allow:
  allowable member stress.
- u_tip_max:
  displacement limit used for interpretation.
- cond_max:
  condition-number threshold used to reject nearly singular designs.
- a_min, a_max, a_init:
  lower bound, upper bound, and initial cross-sectional area of each member.
- m_vehicle, load_factor, total_lift:
  define the applied load case.
- damage_k:
  severity of graph-based damage around a seed node.

Important note on condition number:
- cond_est is computed by the solver in truss_solver.py.
- cond_max is the user-selected threshold here in optimize_v4_connected.py.
- If cond_est > cond_max, the structure is treated as too close to a mechanism.

Typical things to edit in this file:
- switch lattice family,
- change the wing load,
- change the material properties,
- change the initial member size,
- change the damage severity,
- tighten or loosen cond_max,
- enable or disable grouped pruning,
- control how long the run lasts.

2. truss_solver.py
------------------
This file contains the structural solver.

Primary responsibilities:
- assemble the global truss stiffness matrix,
- apply clamped boundary conditions,
- solve for nodal displacements,
- compute axial member forces,
- compute axial member stresses,
- estimate the condition number of the reduced stiffness matrix,
- define how tip displacement is measured.

Key functions:
- assemble_truss_stiffness(...)
  Builds the global truss stiffness matrix from node coordinates, connectivity,
  cross-sectional areas, and material stiffness.
- apply_dirichlet_bc(...)
  Applies fixed boundary conditions by extracting free DOFs.
- estimate_condition_number(Kff)
  Estimates the condition number of the reduced stiffness matrix.
- choose_tip_node(...)
  Chooses which node is used as the wing-tip response point.
- solve_truss(...)
  Runs the complete truss solution and returns displacements, forces, stresses,
  tip displacement, and cond_est.

Important note:
This is an axial truss model only. Members carry axial load only.
There is no beam bending, torsion, or shell behavior in this model.
That means connected graphs can still behave like mechanisms if they are not
sufficiently triangulated.

3. generate_square_lattice.py
-----------------------------
This file generates the square wing-box lattice.

Primary responsibilities:
- create structured nodes inside the wing-box volume,
- connect them with spanwise, chordwise, and vertical members,
- optionally add diagonals in xy, yz, and xz planes.

This is the main interpretable baseline generator.

Why it matters:
The square lattice is easier to understand than the octet lattice, so it is
better for debugging:
- load paths,
- pruning behavior,
- and damage location.

Important generator parameters:
- ny:
  number of spanwise stations.
- nx:
  number of chordwise stations.
- nz:
  number of depth layers.
- add_xy_diagonals:
  in-plane diagonals on chord-span faces.
- add_yz_diagonals:
  diagonals on span-depth faces.
- add_xz_diagonals:
  diagonals on chord-depth faces.

Interpretation:
- spanwise members behave like spar/rail elements,
- chordwise members behave like ribs,
- vertical members connect upper and lower layers,
- diagonals provide shear bracing and prevent mechanisms.

4. generate_octet_lattice.py
----------------------------
This file generates the denser octet-style lattice.

Primary responsibilities:
- generate a more complex 3D lattice with many diagonal members,
- provide an alternative graph family to compare against the square lattice.

Why it is currently secondary:
The octet graph is much harder to visualize and debug, so it is less useful
for early-stage workflow development.

Use case:
After the workflow is trusted on the square lattice, this generator can be
reactivated to study more complex graph families.

5. damage_models.py
-------------------
This file handles structural damage in graph form.

Primary responsibilities:
- remove nodes and incident edges around a chosen seed node,
- keep only the root-connected component after damage,
- apply graph-screening cleanup,
- return the damaged graph for structural evaluation.

Key functions:
- damage_by_khop(...)
  Removes nodes within k graph hops of the seed node.
- root_connected_component(...)
  Keeps only the component connected to the clamped wing root.
- peel_low_degree_nodes(...)
  Removes non-root nodes with degree below a threshold as a graph-screening step.
- filter_to_root_connected_intact(...)
  Used after pruning to keep only the intact graph component connected to root.
- damage_and_check_full_connectivity(...)
  Main damage routine combining removal, root filtering, and graph screening.

Important conceptual warning:
graph_screen_passed does NOT prove structural stability.
It only means the graph passed the heuristic screening.
Real stability is determined only after solve_truss(...) succeeds and cond_est
is acceptable.

6. gt_metrics.py
----------------
This file computes graph-theoretic metrics.

Primary responsibilities:
- build a NetworkX graph,
- check graph connectivity,
- compute algebraic connectivity,
- compute edge betweenness statistics.

Key functions:
- is_connected_safe(...)
  Returns whether the graph is connected.
- algebraic_connectivity_safe(...)
  Computes lambda_2 if the graph is connected.
- edge_betweenness_stats(...)
  Returns summary statistics of edge betweenness centrality.

Use in the workflow:
These metrics are currently used for debugging and interpretation.
Later they may become:
- pruning guides,
- topology-screening surrogates,
- or optimization constraints/objectives.

7. grouped_pruning.py
---------------------
This file contains grouped topology-modification logic.

Primary responsibilities:
- classify square-lattice members by family,
- build pruning groups,
- score candidate groups for removal,
- remove one group at a time.

Why this file exists:
Single-edge pruning was too fragile and often produced mechanisms immediately.
Grouped pruning is more physically interpretable and easier to debug.

Key functions:
- classify_square_lattice_edges(...)
  Separates spanwise, chordwise, vertical, and diagonal members.
- edge_midpoints(...)
  Computes midpoint coordinates for each edge.
- build_spanwise_bay_groups(...)
  Builds pruning candidates, currently based on diagonal members grouped by
  spanwise bins.
- evaluate_group_scores(...)
  Scores pruning groups by low stress, low force, and low area.
- prune_one_group(...)
  Removes a chosen group of members.

Current limitation:
The present grouped pruning only targets diagonal groups.
Later versions may also define groups for:
- ribs,
- spars,
- vertical posts,
- or larger panel families.

8. viz_square_lattice.py
------------------------
This file contains plotting functions for the square lattice.

Primary responsibilities:
- visualize the initial square lattice by member family,
- show orthographic structural views,
- show damage location,
- show removed-node/removed-edge plots,
- show the surviving damaged graph.

Key functions:
- visualize_square_lattice(...)
  Main overview plot for the starting lattice.
- visualize_damage_on_square_lattice(...)
  Shows:
  * original lattice with removed nodes,
  * edges incident to removed nodes,
  * surviving root-connected damaged graph.

Why this file matters:
Without these plots it is very difficult to understand:
- whether damage is meaningful,
- what part of the wing is being removed,
- and whether the surviving graph still represents a plausible structure.

9. load_mapping.py
------------------
This file maps aerodynamic/structural load assumptions to nodal forces.

Primary responsibilities:
- build the nodal force vector for the intact and damaged graph,
- apply a distributed vertical load over the span.

Key functions:
- distributed_vertical_load_to_nodes(...)
  Creates the global force vector for a chosen distribution.
- nodal_force_lookup_table(...)
  Allows manually supplied nodal forces in the future.

Current status:
This is still a simplified load model.
The load is not yet a high-fidelity wing pressure distribution.
Later it can be replaced by:
- beam-derived nodal loads,
- UM/NAST loads,
- or more realistic aerodynamic loading.

10. wingbox_domain.py
---------------------
This file defines the wing-box geometric domain and root mask.

Primary responsibilities:
- store the wing dimensions,
- identify root nodes for clamped boundary conditions.

Key method:
- root_mask(...)
  Returns which nodes belong to the clamped root face.

This file is simple, but it is important because root-connected filtering and
boundary conditions both depend on it.

Current Workflow Summary
========================

The current workflow is:

1. build a lattice graph,
2. assign initial member areas,
3. apply clamped root boundary conditions,
4. solve the intact truss problem,
5. compute stresses, forces, tip displacement, and condition estimate,
6. apply graph-based damage,
7. solve the damaged graph if it passes graph screening,
8. resize members based on stress utilization,
9. attempt grouped pruning,
10. accept pruning only if intact and damaged trial graphs remain feasible,
11. save all results to HDF5,
12. visualize the final or damaged graph.

Why Results May Still Look Unphysical
=====================================

Several current behaviors can still be misleading:

1. Damaged tip response may not be directly comparable to intact response
-----------------------------------------------------------------------
The damaged graph may choose a different effective tip node or load path.
So damaged displacement being smaller than intact does not necessarily mean
damage improved the structure.

2. Truss-only modeling can create mechanisms easily
---------------------------------------------------
Because members carry only axial load, a connected graph can still be nearly
singular if it is not sufficiently triangulated.

3. Damage screening is heuristic
--------------------------------
The current damage model uses graph operations and low-degree peeling before the
solver, but the final authority is still the structural solve.

4. Grouped pruning is still conservative
----------------------------------------
Pruning often gets rejected because even one diagonal-group removal can move the
structure close to a mechanism.

Where to Tune the Main Behavior
===============================

If stresses are too low:
- reduce a_init,
- reduce graph density,
- increase m_vehicle,
- increase load_factor.

If the structure becomes singular:
- add diagonal families,
- increase graph redundancy,
- tighten pruning acceptance,
- reduce damage severity.

If pruning never succeeds:
- reduce graph redundancy,
- enlarge or redefine pruning groups,
- prune only selected diagonal families,
- tighten cond_max acceptance criterion.

Most Important User-Tunable Variables
=====================================

In optimize_v4_connected.py:
- lattice_type
- E_modulus
- density
- sigma_allow
- cond_max
- a_min
- a_max
- a_init
- m_vehicle
- load_factor
- damage_k
- n_iter

In generate_square_lattice.py:
- nx
- ny
- nz
- add_xy_diagonals
- add_yz_diagonals
- add_xz_diagonals

In grouped_pruning.py:
- n_span_bins
- pruning group definition
- pruning score weighting

Recommended Near-Term Development Order
=======================================

1. Keep the square lattice as the main debug baseline.
2. Refine the damaged-response metric so damaged and intact are evaluated at a
   comparable geometric tip region.
3. Tighten pruning acceptance using a lower cond_max threshold for accepted
   pruned designs.
4. Improve grouped pruning units so they reflect meaningful structural bays.
5. Only after that, introduce root-to-tip gradient sizing or graph-density
   grading.
6. After the workflow is trusted, reactivate octet or more complex graph
   families.

Bottom Line
===========
This codebase is now less about "optimization" in the classical sense and more
about building a trustworthy graph-based structural design workflow. The main
driver file coordinates the process, the solver checks real structural response,
the damage model defines candidate failure scenarios, the graph-metric module
provides GT diagnostics, the grouped-pruning file attempts interpretable topology
changes, and the visualization file helps determine whether the graph operations
actually make physical sense.