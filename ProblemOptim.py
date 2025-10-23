"""
/!\ Pymoo VERSION O.6.1.3
path of modules and functions from pymoo may change
depending on the installed version
"""
import numpy as np
import pandas as pd
import os

from _functions_AS_optim import map_to_physical

from pymoo.core.problem import ElementwiseProblem

"""
Objet pour définir le problème d'optim et surtout la fonction _evaluate() qui est appelée par l'algo d'optim, ici on évalue la fonction coût sur MATLAB VOCO
"""

def constraints(X):
    # Retourner un tableau g(x) <= 0 (inégalités)
    # Pour égalité h(x)=0: return [abs(h(x)) - 1e-6, ...]
    return np.array([
        [X[0]-X[1]]   # g1(x) <= 0
    ])

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
            

        # ind_to_optim
        # mid_para : liste des 22 paramètres pris au milieu des plages de valeur pour la categorie choisie
        
        # ---- CONSTRUCTEUR de l'objet Problem_optim_train
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