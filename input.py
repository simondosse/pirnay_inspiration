import numpy as np

class ModelParameters:
    """
    Define the parameters of the model.
    """

    def __init__(self,s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t, model_aero = 'Theodorsen'):
        ''' 
        Initialize the model parameters.

        Parameters:
        -----------
        s : float
            Half-span of the wing (m)
        c : float
            Chord of the wing (m)
        x_ea : float
            Elastic axis location from leading edge (m)
        x_cg : float
            Inertial axis location from leading edge (m)
        m : float
            Mass per unit length (kg/m)
        I_alpha : float
            Torsional inertia per unit length (kg*m)
        EIx : float
            Bending stiffness (N*m^2)
        GJ : float
            Torsional stiffness (N*m^2)
        eta_w : float
            Bending damping ratio
        eta_alpha : float
            Torsional damping ratio
        Mt : float or None
            Wingtip mass (kg). If None, no wingtip mass is considered.
        I_alpha_t : float or None
            Wingtip torsional inertia (kg*m). If None, no wingtip inertia is considered.
        x_t : float or None
            Wingtip location from leading edge (m). If None, no wingtip location is considered.
        model_aero : str
            Aerodynamic model to use ('Theodorsen' or 'QuasiSteady')
        '''
        
        # Geometric parameters
        self.s = s                              # Half-span 
        self.c = c                              # Chord 
        self.b = c / 2                          # Semi-chord

        self.x_ea = x_ea                        # Elastic axis location
        self.x_cg = x_cg                        # Inertial axis location
        self.x_alpha = x_cg - x_ea              # Distance between ea and cg
        self.a = (self.x_ea / self.b) - 1       # Non-dimensional ea location

        # Structural parameters
        self.m = m                              # Mass per unit length
        self.I_alpha = I_alpha                  # Torsional inertia per unit length
        self.EIx = EIx                          # Bending stiffness 
        self.GJ = GJ                            # Torsional stiffness
        self.eta_w = eta_w                      # Bending damping ratio
        self.eta_alpha = eta_alpha              # Torsional damping ratio

        # Wingtip parameters
        if Mt is not None:
            self.has_tip = True                 # Flag to indicate presence of wingtip mass
            self.Mt = Mt                        # Wingtip mass
            self.I_alpha_t = I_alpha_t          # Wingtip torsional inertia
            self.x_t = x_t                      # Wingtip location
        
        else:
            self.has_tip = False
            self.Mt = 0
            self.I_alpha_t = 0
            self.x_t = 0

    
        # Discretization parameters
        self.dy = 1/1000                        # Spatial discretization step
        self.y = np.arange(0, s, self.dy)       # Spatial discretization vector
        self.Nw = 3                             # Number of bending modes
        self.Nalpha = 3                         # Number of torsional modes

        # Aerodynamic parameters
        self.rho_air = 1.204                    # Air density
        self.dCn = 4                          # Normal force coefficient
        self.dCm = 0.5                         # Moment coefficient
        # actually the dCm shoudn't be constant, in the stall region the curve is not linear anymore
        self.Umax = 35                          # Maximum velocity of the IAT wind tunnel
        self.steps = 100                        # Number of velocity steps
        self.U = np.linspace(0.1, self.Umax, self.steps)

        if model_aero not in ['Theodorsen', 'QuasiSteady']:
            raise ValueError("model_aero must be either 'Theodorsen' or 'QuasiSteady'")
        
        self.model_aero = model_aero            # Aerodynamic model to use



    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Parameter has no attribute '{key}'")
