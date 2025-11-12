"""
/!\ Pymoo VERSION O.6.1.3
path of modules and functions from pymoo may change
depending on the installed version
"""
import numpy as np
import pandas as pd
import os
import plotter
import NACA
import ROM

from _functions_AS_optim import map_to_physical
from input import ModelParameters

from pymoo.core.problem import ElementwiseProblem


def constraints(X):
    # Retourner un tableau g(x) <= 0 (inégalités)
    # Pour égalité h(x)=0: return [abs(h(x)) - 1e-6, ...]
    return np.array([
        [X[0]-X[1]]   # g1(x) <= 0
    ])

def cost(X,target_mode_idx):
    '''
    Cost function to evaluate Uc for a given set of physical parameters X for each individual for the optimization algorithm

    Parameters
    ----------
    X : array (np_para,) with columns [x_ea/c, x_cg/c, EI, GJ]
        set of physical parameters to evaluate
    target_mode_idx : int
        index of the target mode for which we want to evaluate Uc

    Returns
    -------
    Uc : float
        critical velocity for the target mode
        (can be a real Uc because it crosses,
        extrapolated if it doesnt cross but a negative slope exists, or arbitrary high if no crossing and no negative slope)
    '''
    model = ModelParameters(s=2, c=0.2, x_ea=X[0]*0.2, x_cg=X[1]*0.2, m=2.4, EIx=X[2], GJ=X[3],model_aero='Theodorsen')
    model.Umax = 40
    model.Ustep = 80
    # we rebuild the model with the new physical parameters X each time (a bit long but ok for now)

    # model.airfoil.x_ea = X[0]*c # *c because we are dealing with adimensionnal parameter
    # model.airfoil.x_cg = X[1]*c # peut faire en sorte que quand on appelle model.x_cg ça appelle en fait model.airfoil.x_cg ? moue c'est mieux de laisser comme ça : on comprend + ce que l'on fait
    # model.EIx = X[2]
    # model.GJ = X[3]

    _, damping, *_ = ROM.ModalParamDyn(model)
    
    Uc, _ = ROM.obj_evaluation(U = model.U, damping = damping[:,target_mode_idx]) # /!\ la matrice de damping regarde tous les modes 0,1,2,3
    # plotter.plot_modal_data_single(f,damping,model, suptitle=f'EIx = {model.EIx:.1f}, GJ = {model.GJ:.1f}, x_ea = {model.airfoil.x_ea:.3f}, x_cg = {model.airfoil.x_cg:.3f}, Uc = {Uc:.1f}')
    '''
    the big problem is that Uc extrapolated can be the lowest Uc, but if we extrapolated that means we don't have a damping cross so we want to avoid that
    '''
    return Uc

class ProblemOptim(ElementwiseProblem): # ElementwiseProblem est une sous-classe de Problem qui serait plus simple à utiliser
    def __init__(self, n_var,n_obj,n_ieq_constr,xl,xu,target_mode_idx):
        
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
        self.target_mode_idx = target_mode_idx

        # Lancement de l'algorithme d'optimisation
        # minimize renvoie un :class:`~pymoo.core.result.Result`

    def _evaluate(self, X, out, *args, **kwargs):
        X_physical = map_to_physical(X) # X_physical = [x_ea/c, x_cg/c, EI, GJ]
        out["F"] = cost(X_physical, self.target_mode_idx)
        # out["G"] = constraints(X)   # <= 0