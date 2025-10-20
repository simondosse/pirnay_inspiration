# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:41:38 2025

@author: ADJ
"""

# %% Librairies

import pickle
import numpy as np
import copy
import sys

# sys.path.append("L:/APA/Code_optim_repo_work")
sys.path.append("L:/ZDO/_Python/electric-motor-optimization")
import Global_parameters_sensitivity as gps


import matplotlib.pyplot as plt

# %% Fonction pour le dépouillement

def get_sensitivity_plot_data(results_sensitivity,
                              list_of_parameters_names,
                              objective_index = 0):
    """
    Fonction pour récupérer le nécessaire afin de tracer les résultats de
    sensibilité, pour un objectif précis.

    Parameters
    ----------
    results_sensitivity : list
        Liste de dictionnaires obtenues après exécution de :
            results_sensitivity = gps.recompute_Morris_sensitivity(optim_params, n_levels,
                                                                   allsamples_Morris,
                                                                   function_values_Morris)
        Avec en clefs pour chaque objectif
            # ["mu"] : array of size (n_params,)
            # ["mu_star"] : array of size (n_params,)
            # ["mu_star_conf"] : list of len(n_params)
            # ["names"] : list of len(n_params)
            # ["sigma"] : array of size (n_params,)
            # ["elementary_effects"] : array of size (n_params, n_trajectories)
    
    list_of_parameters_names: list de strings
        La liste des noms des paramètres dont la sensibilité a été analysée
    
    objective_index: int
        Indice de l'objectif
    
    Returns
    -------
    reordered_parameters_names: list de strings
        Les noms de paramètres par ordre décroissant de mu_values
    y_values_to_position_labels: list
        Liste des valeurs en Y pour les tracés de chaque paramètre (dans l'ordre décroissant des mu_values)
    x_axis_mu_values: list
        La liste des valeurs moyennes absolues (dans l'ordre décroissant)
    x_axis_sigma_values: list
        La liste des écarts-types correspondants (dans l'ordre décroissant des mu_values)
    Si possible
        x_elementary_effects: list of arrays de tailles (n_trajectories)
            Pour chaque paramètre, l'array des valeurs de toutes les trajectoires
            Dans l'ordre décroissant des mu_values
        y_elementary_effects
            Pour chaque paramètre, un array de n_valeurs identiques pour la position en ordonnées
    """
    
    #Check de début sur le nombre de paramètres
    n_params = len(results_sensitivity[objective_index]["mu_star"])
    if len(list_of_parameters_names)!= n_params:
        raise TypeError('{} noms de paramètres spécifiés contre {} évalués'.format(len(list_of_parameters_names),
                                                                                   len(n_params)))
    #On récupère la valeur moyenne absolue de l'effet du paramètre
    mu_values = results_sensitivity[objective_index]["mu_star"]
    #On récupère l'écart type de l'effet du paramètre
    sigma_values = results_sensitivity[objective_index]["sigma"]
    #On peut récupérer les paramètres dans l'ordre donné des valeurs précédentes
    #   mais ce doit être la même liste que 'list_of_parameters_names'
    # params_names = results_sensitivity[objective_index]["names"]
    
    #On va classer les paramètres par ordre décroissant de mu_values
    #en passant par un dictionnaire
    dict_to_sort_parameters = {}
    for p in range(n_params):
        dict_to_sort_parameters[list_of_parameters_names[p]] = mu_values[p] #'PARAM_NAME': mu_value
    #On fait le tri sur les clefs suivant leur valeur, dans l'ordre décroissant
        #Fonctionnalité disponible avec Python 3.7+
        dict_of_sorted_parameters = {k: v for k, v in sorted(dict_to_sort_parameters.items(), key=lambda item: item[1], reverse=True)}
        #ne pas de fier à l'ordre d'affichage des clefs dans l'explorateur de variables
    
    #Maintenant que les paramètres sont rangés par mu_star décroissant
    reordered_parameters_names = list(dict_of_sorted_parameters.keys())
    #On récupère la liste des indices initiaux de ces paramètres dans list_of_parameters_names
    reordered_indexes = []
    for p in range(n_params):
        reordered_indexes.append(list_of_parameters_names.index(reordered_parameters_names[p]))
    
    #On retourne alors "le nécessaire" pour les graphs
    x_axis_mu_values = [] #données en horizontal
    x_axis_sigma_values = [] #données en horizontal
    y_values_to_position_labels = sorted(list(range(n_params)), reverse=True)
        # list(range(n_params)) = [0, 1, 2, 3, ..., n_params]
        # sorted(list(range(n_params)), reverse=True) = [n_params, ..., 3, 2, 1, 0]
            # Pour avoir les valeurs maxi de mu_value en haut du graph
    for p in range(n_params):
        #Pour le tracé, on liste les valeurs dans l'ordre des noms
        x_axis_mu_values.append(mu_values[reordered_indexes[p]])
        x_axis_sigma_values.append(sigma_values[reordered_indexes[p]])
    
    #Si on peut récupérer tous les résultats
    if "elementary_effects" in list(results_sensitivity[objective_index].keys()):
        elementary_effects = results_sensitivity[objective_index]["elementary_effects"]
            # Array de taille (n_params, n_trajectories)
            # Des effets elementaires de chaque paramètre sur l'objectif
        #Le nombre de trajectoires évaluées :
        n_trajectories = np.shape(elementary_effects)[1]
        
        #On met en forme ce qu'il faut pour le tracé, en abscisse X et ordonnée Y
        x_elementary_effects = []
        y_elementary_effects = []
        for p in range(n_params): #Pour chaque paramètre
            param_index = reordered_indexes[p] #Par mu_star décroissant
            y_elementary_effects.append([y_values_to_position_labels[p] for k in range(0, n_trajectories)]) #Une liste de y constants, pour positionner les points en horizontal
            x_elementary_effects.append(elementary_effects[param_index,:]) #Le valeurs de chaque effet

        return reordered_parameters_names, y_values_to_position_labels, x_axis_mu_values, x_axis_sigma_values, x_elementary_effects, y_elementary_effects
    
    else:
        return reordered_parameters_names, y_values_to_position_labels, x_axis_mu_values, x_axis_sigma_values


# # %% section MAIN

# if __name__ == "__main__":

#     save_figures = False
    
#     # %% Résultats calculs pour analyse de sensibilté
    
#     path_L = 'L:/MJE/2882_001_e2drives/14_SA_M1_simplifie_more_samples/'

#     # Résultats répartis dans plusieurs dictionnaires
#     files = ["dico_results_12_at_iteration_0_20250125-191453.pickle",
#              "dico_results_34_at_iteration_0_20250125-203254.pickle",
#              "dico_results_56_at_iteration_0_20250125-215206.pickle",
#              "dico_results_78_at_iteration_0_20250125-231036.pickle",
#              "dico_results_910_at_iteration_0_20250126-002738.pickle",
#              "dico_results_1120_at_iteration_0_20250126-065447.pickle",
#              "dico_results_2130_at_iteration_0_20250126-132242.pickle"]
    
#     F = np.array([]) # critères sous forme d'array
#     X = np.array([])
    
    
#     # On récupère les résultats
#     list_of_dico_results = []
#     for k, file in enumerate(files):
#         with open(path_L + file, "rb") as fp:
#             dico = pickle.load(fp)
#             if k > 0:
#                 F = np.concatenate((F, np.transpose(np.array(copy.deepcopy(dico["F"])))), axis=0)
#                 # F : array de taille (n_designs, n_criteres)
#                 X = np.concatenate((X, np.array(copy.deepcopy(dico["X"]))), axis=0) # array de taille (n_designs, n_params)
#                 list_of_dico_results.append(copy.deepcopy(dico))
#             else:
#                 # 1 ère itération
#                 F = np.transpose(np.array(copy.deepcopy(dico["F"]))) # array de taille (n_designs, n_criteres)
#                 X = np.array(copy.deepcopy(dico["X"])) # array de taille (n_designs, n_params)
#                 list_of_dico_results.append(copy.deepcopy(dico))
#             fp.close()
    
#     # Nombre total de designs testés
#     n_designs_all = X.shape[0]
    
#     #### Résultats issus du design initial
    
#     F0_path = "L:/ADJ/e2drive/6-sensitivity_analysis_FINAL/Results_sensitivity_design_initial_runup.pickle"
    
#     with open(F0_path, "rb") as fp:
#         dict_initial_design = pickle.load(fp)
    
#     F0 = list(dict_initial_design.values())
    
    
#     F_reduction = F - F0
#     F_reduction[:, :] = F_reduction[:, :] / F0[:] * 100
    
#     F0 = np.array(F0)
    
    
#     # #### Changement d'unités
#     # # On connaît les critères calculés (et les unités) :
#     #     # 0 : amplitude contribution EM {10, -2} sur WP1 en [MPa]
#     #     # 1 : amplitude contribution EM {10, -2} sur WP2 en [MPa]
#     #     # 2 : couple moyen sur WP1 en [N.m]
#     #     # 3 : couple moyen WP2 en [N.m] 
        
#     # # Passage de [MPa] à dB ref 1 MPa
#     F[:, 0] = 20*np.log10(F[:, 0])
#     F[:, 1] = 20*np.log10(F[:, 1])
#     F[:, 2] = 20*np.log10(F[:, 2])
#     F[:, 3] = 20*np.log10(F[:, 3])
    

    
#     # %% Paramètres
    
#     flux_parameters, variables_types, variables_units,\
#            sensitivity_params, bounds = define_parameters()
           
#     # %% Dépouillement sensibilité
    
#     #NOTES Parpinou
#         # allsamples_Morris : array de l'ensemble des jeux de paramètres évalués (=X ici)
#         # function_values_Morris : liste avec pour chaque objectif, array de la fonction évalué pour le jeu de paramètre
#         # optim_params :: liste des noms des paramètres
        
#     n_levels = 6 # Niveaux définis pour les valeurs prises par les paramètres
    
#     function_values_Morris = [] # init
#     for index in range(F.shape[1]):
#         function_values_Morris.append(copy.deepcopy(F[:, index]))
#     # function_values_mMorris
#         # Liste de taille (n_criteres=4)
#         # Chaque élement de la la liste est un array numpy de taille (n_designs=40,)
    
    
    
#     #Analyse de sensibilité d'après les résultats récupérés
#     results_sensitivity = gps.recompute_Morris_sensitivity(sensitivity_params, n_levels,
#                                                            X,
#                                                            function_values_Morris)
#     # results_sensitivity = liste de dictionnaires
#         # Avec pour chaque critère évalué
#             # ["mu"] : array of size (n_params,)
#             # ["mu_star"] : array of size (n_params,)
#             # ["mu_star_conf"] : list of len(n_params)
#             # ["names"] : list of len(n_params)
#             # ["sigma"] : array of size (n_params,)
#             # ["elementary_effects"] : array of size (n_params, n_trajectories)         
           
#     # %% TOUS LES GRAPHS
    
#     n_params = len(sensitivity_params)
    
#     #Données pour tous les graphs
#     A = []
#     for objective_index in range(F.shape[1]):
#         A.append(get_sensitivity_plot_data(results_sensitivity,
#                                            sensitivity_params,
#                                            objective_index=objective_index))

#     #Labels...
#     xlabels = ["EM contribution amplitude variation in dB",
#                "EM contribution amplitude variation in dB",
#                "EM contribution amplitude variation in dB",
#                "EM contribution amplitude variation in dB",
#                "Mean torque variation in N.m",
#                "Mean torque variation in N.m"]
#     titles = ["Sensitivity results for EM contribution (10, -2)\nWP1",
#               "Sensitivity results for EM contribution (10, -2)\nWP2",
#               "Sensitivity results for EM contribution (20, 2)\nWP1",
#               "Sensitivity results for EM contribution (20, 2)\nWP2",
#               "Sensitivity results for Mean torque\nWP1 - nominal torque = 1.2 N.m",
#               "Sensitivity results for Mean torque\nWP2 - nominal torque = 2.3 N.m"
#               ]
#     units = ['dB', 'dB','dB','dB', 'N.m', 'N.m']
#     xlims = [(-15, 15), (-15, 15),(-15, 15),(-15, 15), (-30, 30), (-50, 50)]
#     xlims = [None, None, None, None, None, None]
#     #plot_names = ["sensibilite_objectif_1_maxw75.png",
#                   #"sensibilite_objectif_2_maxw100.png",
#                   #"sensibilite_objectif_3_mt75.png",
#                   #"sensibilite_objectif_4_mt100.png"
#                   #]
    
#     #Tracé
    
#     for k, a in enumerate(A):
#         plt.figure(figsize=(10, 8))

#         #Line at 0
#         plt.axvline(0, color='black', linewidth=1.0)
#         n_trajectories = len(a[4][0])
#         for param_index in range(0, n_params):
#             if param_index > 0:
#                 plt.scatter(a[4][param_index], a[5][param_index], c = 'black')
#             else: #One plot with legend
#                 plt.scatter(a[4][param_index], a[5][param_index], c = 'black',
#                             label = 'Elementary effect')
#         # Scatter of mu and sigma
#         plt.scatter(a[2], a[1], s=200, marker="h",c='red',label="Mean absolute value")
#         plt.scatter(a[3], a[1], s=150, marker="*",c='blue',label="Standard deviation")
#         # With annotations for each parameter
#         for param_index in range(0, n_params):
#             plt.annotate("{} ".format(round(a[2][param_index], ndigits=1)) + units[k], (a[2][param_index]*0.8, a[1][param_index] + 0.15), fontsize=10., color='red')

#         # Names of parameters
#         plt.yticks(a[1], a[0], fontsize=14)
#         plt.ylim((-0.5, n_params-0.5)) #Pour ajuster les ordonnées et y voir clair
#         # x label to use
#         plt.xlabel(xlabels[k], fontsize=12)
#         #x limits if specified
#         if xlims[k] is not None:
#             plt.xlim(xlims[k])
            
#         plt.title(titles[k] + "\n\n", fontsize=14) # "\n\n" pour laisser espace pour la légende...
        
#         plt.grid(linestyle="--",linewidth=0.2)
        
#         plt.legend(fontsize = 12.)
#         plt.legend(bbox_to_anchor=(0., 1.), ncol=3, loc='lower left', shadow=True) #, fancybox=True, shadow=True
#         plt.tight_layout()
        
#         if save_figures:
#             plt.savefig(plot_names[k])
#         # plt.close()
        
        
        
        
#     # %% 
    
#     # Dégradé des couleurs suivant le régime considéré
#     import matplotlib as mpl
#     n_cases = 5 # 5 régimes au total, même si 4 ici, pour cohérence couleurs entre graphs
#     norm_all = mpl.colors.Normalize(vmin=0, vmax=n_cases-1)
#     cmap_all = mpl.cm.ScalarMappable(norm=norm_all, cmap=mpl.cm.viridis)     
    
    
    
#     criteria = []
#     titles = ["EM contribution (10,-2)","EM contribution (20,2)"]
    
#     fig, ax = plt.subplots(1,2, figsize=(12,8))

#     # Tous les designs testés
#     for ax_index in range(2):
#         ax[ax_index].grid(color='grey', linestyle='--') 
#         # Lignes à 0
#         ax[ax_index].axhline(0, color='black')
#         ax[ax_index].axvline(0, color='black')
        
#         ax[ax_index].scatter(F_reduction[:,4], #Mean torque WP1
#                              F_reduction[:,0+2*ax_index], #harmonique de couple WP1
#                              color = cmap_all.to_rgba(0),
#                              label = "3500 rpm")
#         ax[ax_index].scatter(F_reduction[:,5], #Mean torque WP2
#                              F_reduction[:,1+2*ax_index], #harmonique de couple WP2
#                               color = cmap_all.to_rgba(2),
#                               label ="4600 rpm")
#         ax[ax_index].set_xlabel('[%]\nMean torque variation from initial design', fontsize=9.)
#         ax[ax_index].set_ylabel('[%]\n' + titles[ax_index] + ' variation from initial design', fontsize=12.)
#         ax[ax_index].set_title(titles[ax_index]+" VS Mean Torque", fontsize=12.)
#         ax[ax_index].legend(fontsize=12)
#         #fig.suptitle(WPs_labels[0] + "\nComparison of tested designs for sensitivity analysis", fontsize=16.)
#         fig.tight_layout()

# # %% Espace de solutions acceptables

# # Recherche des designs où les variations des excitations sont négatives, sans réduction du couple

# indexes_interesting_SA_H10 = []
# indexes_interesting_SA_H20 = []
# indexes_interesting_SA_H10_WP1 = []
# indexes_interesting_SA_H10_WP2 = []
# indexes_interesting_SA_H20_WP1 = []
# indexes_interesting_SA_H20_WP2 = []


# #Objective 1
# for index in range(n_designs_all) :
#     if F_reduction[index,0] < 0: #WP1
#             if F_reduction[index,4] > 0 :         
#                 indexes_interesting_SA_H10.append(index)
#                 indexes_interesting_SA_H10_WP1.append(index)
#     if F_reduction[index,1] < 0 : #WP2
#         if F_reduction[index,5] > 0 :
#             indexes_interesting_SA_H10.append(index)
#             indexes_interesting_SA_H10_WP2.append(index)
                
# X_interesting_SA_H10 = X[list(set(indexes_interesting_SA_H10))]    

# #Objective 2
# for index in range(n_designs_all) :
#     if F_reduction[index,2] < 0: #WP1
#             if F_reduction[index,4] > 0 :
#                 indexes_interesting_SA_H20.append(index)                
#                 indexes_interesting_SA_H20_WP1.append(index)
#     if F_reduction[index,3] < 0 : #WP2
#         if F_reduction[index,5] > 0 :
#             indexes_interesting_SA_H20.append(index)  
#             indexes_interesting_SA_H20_WP2.append(index)

# X_interesting_SA_H20 = X[list(set(indexes_interesting_SA_H20))]
    
# # %% Sensibilité critères par régime
        
# speeds = ["WP1",
#           "WP2"]

# speeds_labels = ["3500m", "4600rpm"]

# objectives_indexes_by_group = [[0,2,4],
#                                [1,3,5]]

# titles_by_group = [["EM contribution (10,-2)", "EM contribution (20,-2)", "Mean torque"],
#                    ["EM contribution (10,-2)", "EM contribution (20,-2)", "Mean torque"]]

# units_by_group = [["dB", "dB", "N.m"],
#                   ["dB", "dB", "N.m"]]
                  
            

# # 1 graph / régime
# for s in range(len(speeds)):
#     fig, ax = plt.subplots(1, len(objectives_indexes_by_group[s]), sharey=True, figsize=(20,10))
#     for x in range(len(objectives_indexes_by_group[s])):
#         ax[x].grid(True, linestyle='--')

#     # Tracé avec des boucles
#     for ax_index, k in enumerate((objectives_indexes_by_group[s])): # Pour chaque critère
#         # ax_index : indice de l'axe pour tracé, pour un groupe de critères
#         # k : valeur = indice de l'objectif
#         if ax_index==0 and k == objectives_indexes_by_group[s][0]:
#             tmp0 = get_sensitivity_plot_data(results_sensitivity, sensitivity_params, objective_index=k)
            
#             # Mu et Sigma
#             ax[ax_index].scatter(tmp0[2], tmp0[1], s=100, marker="h", color="red", label="Mean Elementary effect")
#             ax[ax_index].scatter(tmp0[3], tmp0[1], s=75, marker="*", color="blue", label="Standard deviation")
#             # Elementary Effects
#             for param_index in range(0, n_params):
#                 if param_index > 0:
#                     ax[ax_index].scatter(tmp0[4][param_index], tmp0[5][param_index], s=20, color= "black")
#                 else:
#                     ax[ax_index].scatter(tmp0[4][param_index], tmp0[5][param_index], s=20, color= "black",
#                                           label="Elementary effect")
#             # Avec annotations
#             for param_index in range(0, n_params):
#                 ax[ax_index].annotate("{}".format(round(tmp0[2][param_index], ndigits=1))+
#                                       units_by_group[s][ax_index],
#                                       (tmp0[2][param_index]*0.8, tmp0[1][param_index] + 0.15),
#                                       fontsize=8., color="red")
#         else:
#             # tmp = get_sensitivity_plot_data(results_sensitivity, sensitivity_params, objective_index=k)
#             tmp = get_sensitivity_plot_data(results_sensitivity, sensitivity_params, objective_index=k)
#             # On réorganise paramètres, dans l'ordre du premier graph (du premier critère)
#             tmp = list(tmp)#de tuple à liste
            
#             #Avant de tracer, il remettre dans l'ordre des résultats sur le 1er objectif
#             names_in_tmp0 = tmp0[0] #Les paramètres par ordre décroissant de Mu_values pour le 1er objectif
#             names_in_tmp = tmp[0] #Les paramètres par ordre décroissant de Mu_values pour le 2ème objectif
#             indexes_to_use = [] #to reorder indexes from tmp
#             for param_name in tmp0[0]:
#                 indexes_to_use.append(names_in_tmp.index(param_name))
            
#             #On réorganise c
#             tmp[0] = tmp0[0] #Paramètres
#             tmp[1] = tmp0[1] #Indices initiaux
#             tmp[2] = [tmp[2][index] for index in indexes_to_use] #Mu_values
#             tmp[3] = [tmp[3][index] for index in indexes_to_use] #Star
#             tmp[4] = [tmp[4][index] for index in indexes_to_use] #Effets élémetaires
            
#             # Mu et Sigma
#             ax[ax_index].scatter(tmp[2], tmp[1], s=100, marker="h", color="red")
#             ax[ax_index].scatter(tmp[3], tmp[1], s=75, marker="*", color="blue")
#             # Elementary Effects
#             for param_index in range(0, n_params):
#                 ax[ax_index].scatter(tmp[4][param_index], tmp[5][param_index], s=20, color= "black")
#             # Avec annotations
#             for param_index in range(0, n_params):
#                 ax[ax_index].annotate("{}".format(round(tmp[2][param_index], ndigits=1))+
#                                       units_by_group[s][ax_index],
#                                       (tmp[2][param_index]*0.8, tmp[1][param_index] + 0.15),
#                                       fontsize=8., color="red")
#         #Line at 0
#         ax[ax_index].axvline(0, color='black', linewidth=1.0)

#         ax[ax_index].set_xlabel('Variation in ' + units_by_group[s][ax_index])
#         ax[ax_index].set_title(titles_by_group[s][ax_index], fontsize=12.)
    
#         ax[0].legend(fontsize=8., loc=4, framealpha=0.95) # légende sur premier graph
    
#         # Names of parameters
#         plt.yticks(tmp0[1], tmp0[0], fontsize=12.)
#         ax[0].set_ylim((-0.5, n_params-0.5)) #Pour ajuster les ordonnées et y voir clair
        
#         #Suptitle
#         fig.suptitle('Sensitivity of criteria for ' + speeds[s], fontsize=14.)
        
#         fig.tight_layout()
        
#         if save_figures:
#             plt.savefig("sensitivity_PMSM_%s"%speeds_labels[s])
#     # plt.close()    