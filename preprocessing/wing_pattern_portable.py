from __future__ import annotations
import matplotlib.pyplot as plt
from pattern_project.pattern_generation import area_fraction, generate_vertical_struts
from pattern_project.voronoi_layer import generate_random_voronoi_network
from pattern_project.square_lattice import square_grid_frame_inside
from pattern_project.warped_hex_module import WarpedLatticeParams, generate_warped_lattice_polys
from pattern_project.plotting_pattern import plot_material
from pattern_project.network_export_STEP import export_stacked_polygons_to_step

def generate_polys_for_pattern(pattern: str, Lx: float, Ly: float, phi: float,  w_frame = 0.02):
    """
    Unified dispatcher. All patterns must return (polys, meta, G).
    """
    registry = {
        "square": lambda: square_grid_frame_inside(
            Lx, Ly, p=Ly / 3, phi_target=phi, W_frame = w_frame
        ),

        "original": lambda: generate_vertical_struts(Lx= Lx, Ly=Ly, N=7, W_frame = w_frame),

        "voronoi": lambda: generate_random_voronoi_network(
            Lx=Lx, Ly=Ly, n_points=202, phi_target=phi, seed=5, tol=1e-4, w_frame = w_frame),

        "graded_hexagonal": lambda: generate_warped_lattice_polys(
            WarpedLatticeParams(Lx=Lx, Ly=Ly,s_min=0.05 * Lx, s_max=0.2 * Lx, p=1.5,
                phi_target=phi,
                w_frame=w_frame
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
    w_frame = 0.02 #thicknes of the frame/perimter for each layer
    ribs_Lz = 0.16
    shell_Lz = 0.008
    phi = 0.15 #areal fraction coverage of layer. used only to thicken lines (doesn't modifiy complexity)
    export_step_flag = False
    polys_ribs, _,_ = generate_polys_for_pattern("original", Lx=Lx, Ly=Ly, phi=phi, w_frame=w_frame)

    for pattern in patterns:
        polys, meta, G = generate_polys_for_pattern(pattern, Lx=Lx, Ly=Ly, phi=phi, w_frame=w_frame)
        print(f"[{pattern}] Generated {len(polys)} polygons with info: {meta}")
        density = area_fraction(polys, Lx, Ly)
        print(f"[{pattern}] Achieved area fraction (density): {density:.6f}")
        title = f"{pattern} | φ={density:.4f}"
        plot_material(polys, Lx, Ly, title=title, holes="white", facecolor="black", edgecolor="none")
        if export_step_flag:         # Export to STEP
            layers = [
              (polys, shell_Lz),   # bottom
              (polys_ribs, ribs_Lz), # middle ribs
              (polys, shell_Lz) # top
            ]
            step_filename = f"{pattern}_pattern.step"
            export_stacked_polygons_to_step(layers, step_filename)

    plt.show()