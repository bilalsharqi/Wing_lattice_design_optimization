from __future__ import annotations
import matplotlib.pyplot as plt
from pattern_project.pattern_generation import lattice_params_and_polygons, generate_random_voronoi_network, area_fraction
from pattern_project.warped_hex_module import WarpedLatticeParams, generate_warped_lattice_polys
from pattern_project.plotting_pattern import plot_material
from pattern_project.network_export_STEP import export_polygons_to_step

def generate_polys_for_pattern(pattern: str, Lx: float, Ly: float, phi: float):
    """
    Returns (polys, meta) for a given pattern name.
    Extend this with more patterns as needed.
    """
    if pattern == "square":
        p = Ly / 3
        polys, meta = lattice_params_and_polygons(
            Lx, Ly, p, phi_target=phi, center=True, clip=True
        )
        return polys, meta

    elif pattern == "voronoi":
        n_points = 202
        polys, meta, points, vor = generate_random_voronoi_network(
            Lx=Lx, Ly=Ly, n_points=n_points, phi_target=phi, seed=5, tol=1e-4
        )
        # If you want points/vor later, you can return them too; for now keep same signature.
        return polys, meta
    
    elif pattern == "graded_hexagonal":
        params = WarpedLatticeParams(
        Lx=Lx, Ly=Ly,
        s_min=0.05 * Lx, s_max=0.2*Lx,  
        p = 1.5,
        organic_amp_frac=0.00,
        phi_target=phi,
        )
        
        polys, meta = generate_warped_lattice_polys(params)
        return polys, meta

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

if __name__ == "__main__":
    patterns = ["square","graded_hexagonal", "voronoi"]  # current available patterns
    Lx = 4.3
    Ly = 1.0
    Lz = 0.16
    phi = 0.15 #areal fraction coverage. used only to thicken lines (doesn't modifiy complexity)
    export_step_flag = False

    for pattern in patterns[:2]:
        polys, meta = generate_polys_for_pattern(pattern, Lx=Lx, Ly=Ly, phi=phi)
        print(f"[{pattern}] Generated {len(polys)} bar polygons with info: {meta}")

        density = area_fraction(polys, Lx, Ly)
        print(f"[{pattern}] Achieved area fraction (density): {density:.6f}")

        # Plot each pattern 
        title = f"{pattern} | φ={density:.4f}"
        plot_material(polys, Lx, Ly, title=title, holes="white", facecolor="black", edgecolor="none")

        # Export to STEP
        if export_step_flag == True:
            step_filename = f"{pattern}_pattern.step"
            export_polygons_to_step(
                polygons = polys,
                z_thickness= Lz,
                step_path = step_filename,
        )
    plt.show()
