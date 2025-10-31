#%%
import numpy as np
from typing import Optional

'''
Dataclass previously used to return results has been removed as requested.
All computed properties now live as attributes on the NACA object itself.
'''


class NACA:
    """
    This object is used to described the section used for the airfoil, it manages the 2D airfoil properties
    """

    def __init__(self, c: float, x_ea, x_cg, m, t_c: float = 0.15):

        # Core geometric attributes
        self.c = float(c)            # chord length [m]
        self.b = self.c/2            # semi-chord length
        self.t_c = float(t_c)        # thickness-to-chord ratio (e.g. 0.15 for NACA0015)

        '''
        I put "_" before the attributs name just to say they have a @property @setter
        '''

        # Additional attributes requested
        self._x_ea = float(x_ea)  # elastic axis x-position [m]

        # Use internal storage for x_cg to allow a property with recompute side-effects
        self._x_cg = float(x_cg)  # centroid x-position [m]
        self.x_alpha = self._x_cg-self._x_ea # J'AURAIS PU DEJA ECRIRE self.x_alpha = self.x_cg - self.x_ea car j'ai les @property plus bas 
        self.a = (self._x_ea / self.b) - 1  
        # Mass properties inputs
        self._m = float(m)          # linear mass [kg/m]

        # Discretization and extrusion span used for inertia integration
        self.N = 4000

        self.x = np.linspace(0,self.c,self.N)
        self.h = self.naca00xx_half_thickness(self.x)

        # Initialize all inertia-related outputs to None until computed
        self.A = None
        self.Ix_area = None
        self.Iy_area = None
        self.Ialpha_area_CG = None # moment polaire d'air au CG, c'est bien la somme de Ix_area+Iy_area calcilé autours des axes passant par le CG
        self.mu = None
        self.Ix_mass = None
        self.Iy_mass = None
        self.Ialpha_CG = None # inertia moment mass around the CG axis
        self.Ialpha_EA = None # inertia moment mass around the EA axis
        self.inertia_mass_naca0015()



    def naca00xx_half_thickness(self, x: np.ndarray):
        """
        Half-thickness function of a NACA 00xx symmetric airfoil.
        t_c = relative thickness (e.g. 0.15 for NACA0015).
        h(x)
        """
        # NOTE: Logic preserved from the original function
        xc = x / self.c

        h = 5 * self.t_c * self.c * (
            0.2969 * np.sqrt(xc)
            - 0.1260 * xc
            - 0.3516 * xc**2
            + 0.2843 * xc**3
            - 0.1015 * xc**4
        )
        return h

    def inertia_mass_naca0015(self):
        """
        Compute area, centroid, area moments of inertia, and mass moments for a NACA0015.
        - Solid homogeneous section.
        - Mass moments are given per unit span if span=1.0.
        - If both rho and mu are given, mu takes precedence.
        """

        zt = self.h
        t = 2.0 * zt  # total thickness

        # Section area
        A = np.trapezoid(t, self.x)
        self.A = A
        # CDG in the section plane (y_cg = 0 by symmetry)
        x_cg = self._x_cg

        # Area moments of inertia about the centroid
        # Iy: vertical axis (perpendicular to chord)
        self.Iy_area = np.trapezoid(t * (self.x - x_cg) ** 2, self.x)

        # Ix: horizontal axis (parallel to chord)
        # For a vertical strip of height t, local inertia = t^3/12
        self.Ix_area = (2.0 / 3.0) * np.trapezoid(zt ** 3, self.x)

        # Polar moment
        self.Ialpha_area_CG = self.Ix_area + self.Iy_area

        mu_val = float(self._m)
        # Mass (dynamic) moments per unit span
        self.Ix_mass = mu_val * (self.Ix_area / A)
        self.Iy_mass = mu_val * (self.Iy_area / A)
        self.Ialpha_CG = self.Ix_mass + self.Iy_mass
        self.Ialpha_EA = self.Ialpha_CG + self.m*(self.x_alpha**2)

        # Out-of-plane centroid (mid-span for a solid extrusion)





        # no need to return self right ?? just NACA.intertia_mass_naca0015() might be enough
        # return self

    # --- Properties to keep attributes consistent when changed from outside ---

    '''
    quand on va faire NACA.x_cg nous donner la valeur de NACA._x_cg 

    Grâce au setter ensuite, si on fait NACA.x_cg = 5,
    python appelle automatiquement NACA.x_cg(5)

    Un @setter va toujour de paire avec un @property
    '''

    #________________________________________________________________________________#
    @property #permet d'accéder à une méthode comme si c'était un attribut, 
    def x_cg(self):
        return self._x_cg


    @x_cg.setter
    def x_cg(self, value: float) -> None:
        # Update CG and recompute inertias that depend on it
        self._x_cg = float(value)
        self.x_alpha = self._x_cg-self._x_ea
        self.inertia_mass_naca0015() # COULD BE too much just for the x_cg update

    @property
    def x_ea(self):
        return self._x_ea

    @x_ea.setter
    def x_ea(self, value: float) -> None:
        # Update CG and recompute inertias that depend on it
        self._x_ea = float(value)
        self.x_alpha = self._x_cg-self._x_ea
        self.inertia_mass_naca0015()

    @property
    def m(self) -> Optional[float]:
        # Linear mass (alias for mu input)
        return self._m

    @m.setter
    def m(self, value: Optional[float]) -> None:
        # Update linear mass and recompute mass inertias
        self._m = float(value)

        self.inertia_mass_naca0015()
    #________________________________________________________________________________#
    

    #--------------- PLOT FUNCTION
    def plot_naca00xx_section(
        self,
        ax=None,
        show: bool = True,
        annotate: bool = True,
        fill: bool = True,
    ):
        """
        Plot a symmetric NACA 00xx airfoil section and mark the center of gravity.

        Parameters
        ----------
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

        if self.c <= 0:
            raise ValueError("Chord length c must be > 0.")
        if self.N < 10:
            raise ValueError("N is too small; use N >= 100 for a reasonable plot.")

        # x along the chord from LE (0) to TE (c)
        x = self.x

        # Symmetric NACA 00xx half-thickness distribution
        zt = self.h
        t_c = self.t_c

        # Upper and lower surfaces (no camber for 00xx)
        zu = +zt
        zl = -zt
        
        x_cg = self._x_cg # en vrai peut être qu'ici le self.x_cg marche car dcp ça fait appelle au getter (@property)
        x_ea = self._x_ea

        # Prepare axes
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Plot upper and lower surfaces
        ax.plot(x, zu, color='k', lw=1.2)
        ax.plot(x, zl, color='k', lw=1.2)

        # Optional fill for the airfoil shape
        if fill:
            ax.fill_between(x, zl, zu, color='0.85', alpha=0.9, linewidth=0)

        # Mark the CG on the chord axis (y=0 by symmetry)
        ax.plot([x_cg], [0.0], 'go', ms=6, label='CG')
        if annotate:
            ax.annotate('CG', (x_cg, 0.0), xytext=(6, 8),
                        textcoords='offset points', color='g')
            
        ax.plot([x_ea], [0.0], 'ro', ms=6, label='EA')
        if annotate:
            ax.annotate('EA', (x_ea, 0.0), xytext=(6, 8),
                        textcoords='offset points', color='r')

        # Formatting
        tc_percent = int(round(t_c * 100))
        ax.set_title(f'NACA00{tc_percent:02d} section (c = {self.c:g} m)')
        ax.set_xlabel('x [m] (from Leading Edge)')
        ax.set_ylabel('z [m]')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linewidth=0.3, alpha=0.5)
        # ax.legend(frameon=False, loc='best')


        plt.show()



# --- Example usage ---
if __name__ == "__main__":
    #   	 u            v           EIx           GJ
    X_opt = [ 0.30252596, 0.83633956, 249.05822664, 49.63251818]
    #        x_ea         x_cg        EIx           GJ
    X_phys= [ 0.0605    , 0.167     , 249.05822664, 49.63251818]

    airfoil = NACA(x_ea=X_phys[0],x_cg=X_phys[1],m=2.4,c=0.2)
    airfoil.plot_naca00xx_section()

    airfoil.x_ea = 0.05
    print(f'1. Ialpha_EA = {airfoil.Ialpha_EA}')
    print(f'1. x_alpha = {airfoil.x_alpha}')
    airfoil.x_ea = 0.1
    print(f'2. Ialpha_EA = {airfoil.Ialpha_EA}')
    print(f'2. x_alpha = {airfoil.x_alpha}')


# %%
