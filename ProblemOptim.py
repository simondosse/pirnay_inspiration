"""
/!\ Pymoo VERSION O.6.1.3
path of modules and functions from pymoo may change
depending on the installed version
"""
import numpy as np
import pandas as pd
import os

import NACA
import ROM

from _functions_AS_optim import map_to_physical
from input import ModelParameters

from pymoo.core.problem import ElementwiseProblem


# fixed parameters
s, c = 2.0, 0.2
m = 2.4
eta_w = 0.005
eta_alpha = 0.005

def constraints(X):
    # Retourner un tableau g(x) <= 0 (inégalités)
    # Pour égalité h(x)=0: return [abs(h(x)) - 1e-6, ...]
    return np.array([
        [X[0]-X[1]]   # g1(x) <= 0
    ])

def cost(X):
    x_ea = X[0]*c
    x_cg = X[1]*c
    res = NACA.inertia_mass_naca0015(c=c, mu=m, N=4000, span=s, xcg_known=x_cg)
    I_alpha=res.Jz_mass+m*abs(x_cg-x_ea)**2 # parallel axis theorem to get torsional inertia about elastic axis
    EIx = X[2]
    GJ = X[3]

    model = ModelParameters(s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, None, None, None, 'Theodorsen')
    _, damping, *_ = ROM.ModalParamDyn(model)
    Uc, _ = ROM.damping_crossing_slope(U = model.U, damping = damping[:,0])
    return Uc

class ProblemOptim(ElementwiseProblem): # ElementwiseProblem est une sous-classe de Problem qui serait plus simple à utiliser
    def __init__(self, n_var,n_obj,n_ieq_constr,xl,xu):
        
        #Initialisation de l'objet :
        #n_var : nombre de variables d'optimisation
        #n_obj : nombre d'objectifs
        #n_constr : nombre de contraintes
        
        #bnds_low_optim,bnds_up_optim: limites de chaque paramètre,
        #rentrées comme arguments à la définition de l'objet sous la forme : 
        #xl=[bound_low_param_1, ..., bound_low_param_n_var]
        #xu=[bound_up_param_1, ..., bound_up_param_n_var]
        
        # ---- CONSTRUCTEUR de l'objet ProblemOptim
        super().__init__(n_var=n_var, # hérite de certains attributs
                         n_obj=n_obj,
                         n_ieq_constr=n_ieq_constr,
                         xl=xl,
                         xu=xu)
        # self.ind_to_optim = ind_to_optim
        self.count = 0

        # Lancement de l'algorithme d'optimisation
        # minimize renvoie un :class:`~pymoo.core.result.Result`

    def _evaluate(self, X, out, *args, **kwargs):
        X_physical = map_to_physical(X)
        out["F"] = cost(X_physical)
        # out["G"] = constraints(X)   # <= 0