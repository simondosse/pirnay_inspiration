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
# we define a model with default random parameters for the X[i], those will be updated during the evaluation F(X)
model = ModelParameters(s, c, x_ea=0.05, x_cg=0.13, m=m, EIx=400, GJ=70, eta_w=eta_w, eta_alpha=eta_alpha, model_aero='Theodorsen')
def constraints(X):
    # Retourner un tableau g(x) <= 0 (inégalités)
    # Pour égalité h(x)=0: return [abs(h(x)) - 1e-6, ...]
    return np.array([
        [X[0]-X[1]]   # g1(x) <= 0
    ])

def cost(X):
    model.airfoil.x_ea = X[0]*c # *c because we are dealing with adimensionnal parameter
    model.airfoil.x_cg = X[1]*c # peut faire en sorte que quand on appelle model.x_cg ça appelle en fait model.airfoil.x_cg ? moue c'est mieux de laisser comme ça : on comprend + ce que l'on fait
    model.EIx = X[2]
    model.GJ = X[3]

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