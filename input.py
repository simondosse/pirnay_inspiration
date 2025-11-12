import numpy as np
from NACA import NACA

class ModelParameters:
    """
    Define the parameters of the model.
    """

    def __init__(self,s, c, x_ea, x_cg, m, EIx, GJ, eta_w=0.005, eta_alpha=0.005, Mt=None, I_alpha_t=None, x_t=None, model_aero = 'Theodorsen'):
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
        self._s = s                              # Half-span 

        self.airfoil = NACA(c=c,t_c=0.15,m=m,x_ea=x_ea,x_cg=x_cg)
             # Non-dimensional ea location

        # Structural parameters
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
        self._dy = 1/1000                        # Spatial discretization step
        self.y = np.arange(0, s, self._dy)       # Spatial discretization vector
        self._Nv = 0
        self._Nw = 3                             # Number of bending modes
        self._Nalpha = 3                         # Number of torsional modes
        self.Nq = self.Nw+self.Nalpha+self.Nv
        self.nDOF = 2

        # Aerodynamic parameters
        self.rho_air = 1.204                    # Air density

        '''
        the slope of aero coeff should be attribut of NACA ? actually nop if it's 3D coeff
        '''
        self.dCn = 4                          # Normal force coefficient, normalement c'est 2pi pour une aile infinie, pour une aile finie on la correction a0 / (1 + a0/(π e AR))
        self.dCm = 0.5                         # Moment coefficient
        # actually the dCm shoudn't be constant, in the stall region the curve is not linear anymore

        self._Umax = 30                          # Maximum velocity of the IAT wind tunnel
        self._Ustep = 60                        # Number of velocity steps
        self.U = np.linspace(0.1, self._Umax, self._Ustep)

        if model_aero not in ['Theodorsen', 'QuasiSteady']:
            raise ValueError("model_aero must be either 'Theodorsen' or 'QuasiSteady'")
        
        self._model_aero = model_aero            # Aerodynamic model to use



    '''Helpers internes____________________________________________________________________________________________'''
    def _recompute_y(self):
        if hasattr(self, "_s") and hasattr(self, "_dy"):
            self.y = np.arange(0, self._s, self._dy)

    def _recompute_Nq(self):
        # Cohérent avec le code ROM qui utilise Nq = Nw + Nalpha
        if hasattr(self, "_Nw") and hasattr(self, "_Nalpha"):
            self.Nq = int(self._Nv) + int(self._Nw) + int(self._Nalpha)

    def _recompute_U(self):
        if hasattr(self, "_Umax") and hasattr(self, "_Ustep"):
            self.U = np.linspace(0.1, float(self._Umax), int(self._Ustep))

    def _refresh_tip_state(self):
        # has_tip déterminé uniquement par Mt (non None et non nul)
        self.has_tip = (self._Mt is not None) and bool(self._Mt)
        if not self.has_tip:
            self._Mt = 0
            self._I_alpha_t = 0
            self._x_t = 0
        # recopie vers attributs "publics" pour compatibilité
        self.Mt = self._Mt
        self.I_alpha_t = self._I_alpha_t
        self.x_t = self._x_t

    '''@PROPERTY and @SETTER____________________________________________________________________________________________'''
    # --- s ---
    @property
    def s(self) -> float:
        return self._s

    @s.setter
    def s(self, value: float) -> None:
        self._s = float(value)
        self._recompute_y()

    # --- dy ---
    @property
    def dy(self) -> float:
        return self._dy

    @dy.setter
    def dy(self, value: float) -> None:
        self._dy = float(value)
        self._recompute_y()

    # --- Nv ---
    @property
    def Nv(self) -> int:
        return self._Nv

    @Nv.setter
    def Nv(self, value: int) -> None:
        self._Nv = int(value)
        self._recompute_Nq()

    # --- Nw ---
    @property
    def Nw(self) -> int:
        return self._Nw

    @Nw.setter
    def Nw(self, value: int) -> None:
        self._Nw = int(value)
        self._recompute_Nq()

    # --- Nalpha ---
    @property
    def Nalpha(self) -> int:
        return self._Nalpha

    @Nalpha.setter
    def Nalpha(self, value: int) -> None:
        self._Nalpha = int(value)
        self._recompute_Nq()

    # --- model_aero ---
    @property
    def model_aero(self) -> str:
        return self._model_aero

    @model_aero.setter
    def model_aero(self, value: str) -> None:
        if value not in ['Theodorsen', 'QuasiSteady']:
            raise ValueError("model_aero must be either 'Theodorsen' or 'QuasiSteady'")
        self._model_aero = value

    # --- Umax ---
    @property
    def Umax(self) -> float:
        return self._Umax

    @Umax.setter
    def Umax(self, value: float) -> None:
        self._Umax = float(value)
        self._recompute_U()

    # --- steps ---
    @property
    def Ustep(self) -> int:
        return self._Ustep

    @Ustep.setter
    def Ustep(self, value: int) -> None:
        self._Ustep = int(value)
        self._recompute_U()


    '''
    # --- Mt ---
    @property
    def Mt(self):
        return self._Mt

    @Mt.setter
    def Mt(self, value) -> None:
        self._Mt = value if value is not None else None
        self._refresh_tip_state()

    # --- I_alpha_t ---
    @property
    def I_alpha_t(self):
        return self._I_alpha_t

    @I_alpha_t.setter
    def I_alpha_t(self, value) -> None:
        self._I_alpha_t = value if value is not None else 0
        self._refresh_tip_state()

    # --- x_t ---
    @property
    def x_t(self):
        return self._x_t

    @x_t.setter
    def x_t(self, value) -> None:
        self._x_t = value if value is not None else 0
        self._refresh_tip_state()
        '''

