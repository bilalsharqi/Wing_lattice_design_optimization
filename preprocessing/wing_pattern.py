from __future__ import annotations
from pattern_project.pattern_generation import lattice_params_polys_and_graph, generate_random_voronoi_network, area_fraction, generate_vertical_struts
from pattern_project.warped_hex_module import WarpedLatticeParams, generate_warped_lattice_polys
from pattern_project.network_export_STEP import export_stacked_polygons_to_step
from pattern_project.plotting_pattern import plot_material

def generate_polys_for_pattern(pattern: str, Lx: float, Ly: float, phi: float):
    """
    Unified dispatcher. All patterns must return (polys, meta, G).
    """
    registry = {
        "square": lambda: lattice_params_polys_and_graph(
            Lx, Ly, p=Ly / 3, phi_target=phi, clip=True
        ),

        "original": lambda: generate_vertical_struts(Lx= Lx, Ly=Ly, N=7, w = 0.054),


        "voronoi": lambda: generate_random_voronoi_network(
            Lx=Lx, Ly=Ly, n_points=202, phi_target=phi, seed=5, tol=1e-4
        ),

        "graded_hexagonal": lambda: generate_warped_lattice_polys(
            WarpedLatticeParams(
                Lx=Lx, Ly=Ly,
                s_min=0.05 * Lx,
                s_max=0.2 * Lx,
                p=1.5,
                phi_target=phi,
            )
        ),
    }

    try:
        polys, meta, G = registry[pattern]()
    except KeyError:
        raise ValueError(f"Unknown pattern '{pattern}'. Available: {list(registry)}")
    
    G.es['weight'] = list(map(float, G.es['length']))
    return polys, meta, G

if __name__ == "__main__":
    patterns = ["square","voronoi","graded_hexagonal"]
    Lx = 4.3
    Ly = 1.0
    ribs_Lz = 0.16
    shell_Lz = 0.008
    phi = 0.15 #areal fraction coverage. used only to thicken lines (doesn't modifiy complexity)
    export_step_flag = True

    polys_ribs, _,_ = generate_polys_for_pattern("original", Lx=Lx, Ly=Ly, phi=phi)

    for pattern in patterns:
        polys, meta, G = generate_polys_for_pattern(pattern, Lx=Lx, Ly=Ly, phi=phi)
        print(f"[{pattern}] Generated {len(polys)} polygons with info: {meta}")
        density = area_fraction(polys, Lx, Ly)
        print(f"[{pattern}] Achieved area fraction (density): {density:.6f}")
        title = f"{pattern} | φ={density:.4f}"
        plot_material(polys, title = title, Lx = Lx, Ly = Ly)

        if export_step_flag:  # Export to STEP

            layers = [
              (polys, shell_Lz),   # bottom
              (polys_ribs, ribs_Lz), # middle ribs
              (polys, shell_Lz) # top
            ]
            step_filename = f"{pattern}_pattern.step"
            export_stacked_polygons_to_step(layers, step_filename)