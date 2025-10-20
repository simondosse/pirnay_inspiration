import numpy as np
from dataclasses import dataclass
from typing import Optional

'''
thanks to @dataclass we can create a class that only contains data without methods,
it will automatically generate the __init__ method and other methods like __repr__ (for printing)
'''

@dataclass
class InertiaResult:
    # 2D geometry (airfoil section)
    A: float                 # cross-sectional area [m^2]
    x_cg: float              # x-coordinate of the centroid (y_cg = 0 for NACA00xx) [m]
    # 2D area moments of inertia (about the centroid)
    Ix_area: float           # about the horizontal axis through CG (// to chord) [m^4]
    Iy_area: float           # about the vertical axis through CG (⊥ to chord) [m^4]
    Jz_area: float           # polar moment about CG (J = Ix + Iy) [m^4]
    # Linear mass properties
    mu: float                # linear mass [kg/m]
    # 2D mass (dynamic) moments of inertia per unit span
    Ix_mass: float           # [kg·m]
    Iy_mass: float           # [kg·m]
    Jz_mass: float           # [kg·m]
    # Out-of-plane info
    span: float              # extrusion span (thickness/length normal to section) [m]
    z_cg: float              # z-coordinate of CG (mid-span for solid section) [m]

def naca00xx_half_thickness(x: np.ndarray, c: float, t_c: float) -> np.ndarray:
    """
    Half-thickness function of a NACA 00xx symmetric airfoil.
    t_c = relative thickness (e.g. 0.15 for NACA0015).
    """
    xc = x / c
    return 5 * t_c * c * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    )

def inertia_mass_naca0015(
    c: float,
    rho: Optional[float] = None,   # material density [kg/m^3]
    mu: Optional[float] = None,    # linear mass [kg/m] (overrides rho if given), in our case we'll only use mu which m in our ModelParameters class
    N: int = 4000,                 # integration points (≥2000 recommended)
    xcg_known: Optional[float] = None,  # known x_cg position (if already known)
    span: float = 1.0              # extrusion span (out of plane) [m]
) -> InertiaResult:
    """
    Compute area, centroid, area moments of inertia, and mass moments for a NACA0015.
    - Solid homogeneous section.
    - Mass moments are given per unit span if span=1.0.
    - If both rho and mu are given, mu takes precedence.
    """
    if c <= 0:
        raise ValueError("Chord length c must be > 0.")
    if rho is None and mu is None:
        raise ValueError("Provide either rho (density) or mu (linear mass).")
    if N < 500:
        raise ValueError("N too small. Use N >= 2000 for good accuracy.")

    t_c = 0.15  # NACA0015
    x = np.linspace(0.0, c, N)
    yt = naca00xx_half_thickness(x, c, t_c)
    t = 2.0 * yt  # total thickness

    # Section area
    A = np.trapezoid(t, x)

    # CDG in the section plane (y_cg = 0 by symmetry)
    if xcg_known is None:
        x_cg = np.trapezoid(x * t, x) / A
    else:
        x_cg = float(xcg_known)

    # Area moments of inertia about the centroid
    # Iy: vertical axis (perpendicular to chord)
    Iy_area = np.trapezoid(t * (x - x_cg) ** 2, x)

    # Ix: horizontal axis (parallel to chord)
    # For a vertical strip of height t, local inertia = t^3/12
    Ix_area = (2.0 / 3.0) * np.trapezoid(yt ** 3, x)

    # Polar moment
    Jz_area = Ix_area + Iy_area

    # Linear mass
    if mu is None:
        mu_val = rho * A  # solid homogeneous section
    else:
        mu_val = float(mu)

    # Mass (dynamic) moments per unit span
    Ix_mass = mu_val * (Ix_area / A)
    Iy_mass = mu_val * (Iy_area / A)
    Jz_mass = Ix_mass + Iy_mass

    # Out-of-plane centroid (mid-span for a solid extrusion)
    z_cg = span / 2.0

    return InertiaResult(
        A=A,
        x_cg=x_cg,
        Ix_area=Ix_area,
        Iy_area=Iy_area,
        Jz_area=Jz_area,
        mu=mu_val,
        Ix_mass=Ix_mass,
        Iy_mass=Iy_mass,
        Jz_mass=Jz_mass,
        span=span,
        z_cg=z_cg
    )

#--------------- PLOT FUNCTION
def plot_naca00xx_section_with_cg(
    c: float,
    t_c: float = 0.15,
    N: int = 600,
    xcg: Optional[float] = None,
    ax=None,
    show: bool = True,
    annotate: bool = True,
    fill: bool = True,
):
    """
    Plot a symmetric NACA 00xx airfoil section and mark the center of gravity.

    Parameters
    ----------
    c : float
        Chord length [m].
    t_c : float, optional
        Thickness-to-chord ratio (e.g. 0.15 for NACA0015).
    N : int, optional
        Number of x-samples along the chord (resolution).
    xcg : float, optional
        x-position of the CG from the leading edge [m]. If None, compute
        the geometric centroid of the 2D airfoil area (y_cg = 0 by symmetry).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates a new figure and axes.
    show : bool, optional
        If True, call plt.show() at the end.
    annotate : bool, optional
        If True, add a small text label near the CG marker.
    fill : bool, optional
        If True, fill the airfoil shape with a light color.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    if c <= 0:
        raise ValueError("Chord length c must be > 0.")
    if N < 10:
        raise ValueError("N is too small; use N >= 100 for a reasonable plot.")

    # x along the chord from LE (0) to TE (c)
    x = np.linspace(0.0, c, N)

    # Symmetric NACA 00xx half-thickness distribution
    yt = naca00xx_half_thickness(x, c, t_c)

    # Upper and lower surfaces (no camber for 00xx)
    yu = +yt
    yl = -yt

    # Compute CG along chord if not provided (geometric centroid of area)
    if xcg is None:
        t = 2.0 * yt               # local thickness
        A = np.trapezoid(t, x)     # area
        x_cg = np.trapezoid(x * t, x) / A
    else:
        x_cg = float(xcg)

    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot upper and lower surfaces
    ax.plot(x, yu, color='k', lw=1.2)
    ax.plot(x, yl, color='k', lw=1.2)

    # Optional fill for the airfoil shape
    if fill:
        ax.fill_between(x, yl, yu, color='0.85', alpha=0.9, linewidth=0)

    # Mark the CG on the chord axis (y=0 by symmetry)
    ax.plot([x_cg], [0.0], 'ro', ms=6, label='CG')
    if annotate:
        ax.annotate('CG', (x_cg, 0.0), xytext=(6, 8),
                    textcoords='offset points', color='r')

    # Formatting
    tc_percent = int(round(t_c * 100))
    ax.set_title(f'NACA00{tc_percent:02d} section (c = {c:g} m)')
    ax.set_xlabel('x [m] (from Leading Edge)')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linewidth=0.3, alpha=0.5)
    # ax.legend(frameon=False, loc='best')


    plt.show()



# --- Example usage ---
if __name__ == "__main__":
    # Example: chord = 0.3 m, aluminum (rho = 2700 kg/m^3), 1 m span
    c = 0.30
    rho = 2700.0
    res = inertia_mass_naca0015(c=c, rho=rho, N=4000, span=1.0)

    print(f"Area A       = {res.A:.6e} m^2")
    print(f"x_cg         = {res.x_cg:.6e} m (y_cg = 0, z_cg = {res.z_cg:.3f} m for span={res.span} m)")
    print(f"Ix_area      = {res.Ix_area:.6e} m^4")
    print(f"Iy_area      = {res.Iy_area:.6e} m^4")
    print(f"Jz_area      = {res.Jz_area:.6e} m^4")
    print(f"mu (linear mass) = {res.mu:.6e} kg/m")
    print(f"Ix_mass      = {res.Ix_mass:.6e} kg·m")
    print(f"Iy_mass      = {res.Iy_mass:.6e} kg·m")
    print(f"Jz_mass      = {res.Jz_mass:.6e} kg·m")
