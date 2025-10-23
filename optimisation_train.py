# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 08:56:12 2025

@author: ZDO

/!\ Pymoo VERSION O.6.1.3
path of modules and functions from pymoo may change
depending on the installed version
"""
import numpy as np
import pandas as pd
import os

from pymoo.core.problem import ElementwiseProblem
import matlab.engine


"""
Objet pour définir le problème d'optim et surtout la fonction _evaluate() qui est appelée par l'algo d'optim, ici on évalue la fonction coût sur MATLAB VOCO
"""

class Problem_optim_train(ElementwiseProblem): # ElementwiseProblem est une sous-classe de Problem qui serait plus simple à utiliser
    def __init__(self, n_var,n_obj,n_ieq_constr,xl,xu,ind_to_optim,mid_para,eng,basePath,grandparent_folder):
        
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
        self.ind_to_optim = ind_to_optim
        self.mid_para = mid_para
        self.eng = eng
        self.count = 0
        self.basePath = basePath
        self.grandparent_folder=grandparent_folder
        self.csvPath = basePath+"files_csv\\"
        #self.cas = cas
        
    def _evaluate(self, x, out, *args, **kwargs):
        # Il faut lancer la simu MATLAB DEPUIS PYTHON
        # y'a que les para sélectionnés qui changent mais faut renvoyer tous les autres para du veh classique aussi
        # Il faut installer matlabengine pour cela
        
        # On ne va pas rajouter des inputs car sinon faut changer l'endroit où est appelé _evaluate() et ça veut dire changer le library
        
        path_save_X = self.csvPath + "X_to_evaluate.csv"
        # breakpoint()
        #path_save_cas = "L:\\ZDO\\X_cas.csv"
        j=0
        X_full = self.mid_para
        # on stock le tableau sur une autre varibale pour ne pas changer l'attribut de notre objet d'optim,
        # et qu'on se base sur le même veh classique à chaque _evaluate()
        for i in self.ind_to_optim:
            X_full[i] = x[j]
            j+=1
        
        # écriture du fichier .csv récupéré par MATLAB
        np.savetxt(path_save_X, X_full, delimiter=',')
        
        # session MATLAB déjà ouverte, exécution de script depuis Python

        self.eng.addpath(self.basePath, nargout=0)
        self.eng.addpath(self.csvPath, nargout=0)
        self.eng.addpath(os.path.join(self.basePath + "_MATLAB"), nargout=0)
        self.eng.addpath(os.path.join(self.grandparent_folder, "Voco", "main"), nargout=0)
        self.eng.addpath(os.path.join(self.grandparent_folder, "Voco", "vocolin"), nargout=0)
        self.eng.addpath(os.path.join(self.grandparent_folder, "Voco_fonctions_utiles_communes"), nargout=0)
        """
        /!\ même si dans la fontion evaluate_one_MR.m y'a le addpath Voco\main ça va pas marcher car
        le script evaluate_one_MR doit être check en entier avant d'être run depuis python. Donc faut que les path soient ajoutés depuis python
        """
        # print(self.eng.path())
        # breakpoint()
        
        self.eng.run(self.basePath+"_MATLAB\\"+"evaluate_one_MR.m", nargout=0)  # nargout=0 signifie que la fonction ne retourne pas de sortie

        # en soit "evalute_one_MR.m" prend en entrée un fichier .csv et ressort un .csv mais c'est pas la sortie directe en gros,
        # on va lire le .csv après

        # on ferme la session MATLAB, peut être qu'il faudrait l'ouvrir qu'une seule fois
        # mais bon après l'objet eng est pas défini sur toutes les feuilles .py
        
        F = pd.read_csv(self.csvPath + 'OUTvoco_temp.csv').to_numpy(dtype=float).flatten()
        G = pd.read_csv(self.csvPath + 'constraints_temp.csv').to_numpy(dtype=float).flatten()

        out["F"] = F
        out["G"] = G

        self.count += 1
        print('________________________________')
        print(f"Evaluation numéro : {self.count} :")
        print(f"X = {x}")
        print(f"F(X)={F}")
        print(f"G(X)={G}")
        print('________________________________')
        # out["G"] = [g1, g2]
        """
        On pourrait remplir out["F"] et G et cette manière :
            out["F"]=anp.column_stack([obj1,obj2])
        c'est utile si nos objectifs sont vect ?
        obj1 : vecteur 1D de taille n_samples
        obj2 : vecteur 1D de taille n_samples
        anp.column_stack([...]) : assemble tout ça en une matrice de forme (n_samples, n_obj)
        
        
        /!\ faut obligatoirement que out["F"]  -->  soit un array de taille (n_obj,)
        pymoo attend que _evaluate() fournisse exactement n_obj objectifs pour chaque solution x
        """
        """
        pour des performances et une compatibilité optimales, pymoo impose que les résultats soient mis dans le dictionnaire out plutôt que retournés
        c'est pour ça que y'a aucun return
        """




