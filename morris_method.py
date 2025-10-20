"""
Created on Wed Lun  4 08:56:12 2025

@author: ZDO
"""


import numpy as np
from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
import matplotlib.pyplot as plt
# from script3_e2drive_SA_post_process_final_modele_simplifie_more_samples import get_sensitivity_plot_data



def generate_trajectories(problem,nb_lvl=4,nb_traj_opti=6,seed=1):
    
    """
    This function 

    Parameters
    ----------
    path_save : str
        Direction + nom du fichier .csv à sauvegarder.
        Exemple : "L:\\ZDO\\trajectories.csv"
    ext_tot : list
        Liste des plages de valeur pour chaque paramètres
    nb_para_var : int
        Nombre de paramètres qui peuvent varier
    nb_lvl : int
        Nombre de valeur que peut prendre chaque paramètre
    seed : int
        Nombre pour fixer l'aléatoire, comme ça on peut répéter les résultats
        
    Returns
    -------
    trajectories : list
        Liste 2D contenant les trajectoires de variation des paramètres à la
        suite les unes des autres
        
    La fonction écrit aussi un fichier .csv des trajectoires 

    """
    # Génération des trajectoires de Morris
    

    # génération des trajectoires
    N = 10 # nombre de trajectoires générées
    nb_lvl = 4 # nombre de niveau par paramètre, ça va fractionner le niveau
    nb_traj_opti = 6 
    
    trajectories = morris.sample(problem, N, nb_lvl, nb_traj_opti,seed = seed)
    # trajectories contient toutes les tajectoires, une trajectoire contient 10 points dans l'espace de paramètres
    # c'est à dire 10 trains par trajectoires, un SEUL paramètre change entre chaque trajectoire
    

    # np.savetxt(path_save, trajectories, delimiter=',')
    # print(f"Le fichier .csv des trajectoires a été sauvegardé :")
    # print('['+path_save+']')
    
    print('Morris trajectories have been created')
    return trajectories


def resultat(problem, trajectories, out, nb_lvl, title, cas, basePath='L:\\ZDO\\', split_graph=False,save=False):
    
    '''
    Fonction pour 
    - traiter les résultats des simu à partir de morris_analyze.analyze 
    - tracer les graphs suivants :
        barres plot mu*
        sigma = f(mu*)
    
    Parameters
    ----------
    problem : dict
        définit le problème d'AS
    trajectories : np.array 2D
        trajectoires de Morris
    out : np.array
        valeurs des objectifs
    nb_lvl : int
        nombre de niveau que peut prendre chaque paramètre, 4 dans la littérature
    title : str
    
    
    Returns
    ----------
    results
    elementary_effect
    '''
    # On traite les résultats
    results = morris_analyze.analyze(problem, trajectories, out, print_to_console=True, num_levels=nb_lvl)

    # --- Figure 1 : scatter mu_star / sigma ---
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(results['mu_star'], results['sigma'], c='b')
    for i, name in enumerate(problem['names']):
        ax1.text(results['mu_star'][i], results['sigma'][i], f"${name}$")
    ax1.set_xlabel(r"$\mu_{EE}^*$")
    ax1.set_ylabel(r"$\sigma_{EE}$")
    ax1.set_title(title + ' ' + cas)
    fig1.tight_layout()
    plt.show()
    
    name1 = basePath + '_Python\\save_plot_morris\\' +cas+'\\'+ f'scatterPlot_MuSigma_{title}_{cas}'
    if save:
        fig1.savefig(name1)

    # --- Figure 2 : barplot des mu_star triés ---
    sorted_indices = np.argsort(results['mu_star'])[::-1]
    sorted_mu_star = np.array(results['mu_star'])[sorted_indices]
    sorted_names = np.array(problem['names'])[sorted_indices]
    sorted_names_latex = [f"${name}$" for name in sorted_names]

    nb_para = len(sorted_mu_star)
    name2 = basePath + '_Python\\save_plot_morris\\' +cas+'\\'+ f'barPlot_Sigma_{title}_{cas}'
    
    # si on a trop de paramètres on peut tout split en 2 graph
    if split_graph==False and nb_para > 24 :
        split_graph = True
        
    if split_graph:
        split_indices =[
            (0, nb_para // 2), 
            (nb_para // 2, nb_para)
        ]

        suffixes = ['_part1', '_part2']
        
        xmin, xmax = None, None

        for (start, end), suffix in zip(split_indices,suffixes):
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.barh(sorted_names_latex[start:end], sorted_mu_star[start:end], color='blue')
            ax.set_xlabel(r"$\mu_{EE}^*$")
            ax.set_ylabel('Paramètres')
            ax.tick_params(axis='y', labelsize=11)
            ax.set_title(f"{title} {cas} {suffix.replace('_', ' ')}")
            fig.subplots_adjust(left=0.4)
            fig.tight_layout()

            if suffix == '_part1':
                xmin, xmax = ax.get_xlim()
            else:
                ax.set_xlim(xmin, xmax)

            if save:
                fig.savefig(name2 + suffix)
                #fig.savefig(name_base + suffix, dpi=300) pour une meilleure qualité si jamais
            plt.show()
    else:
        fig2, ax2 = plt.subplots(figsize=(6, 8))
        ax2.barh(sorted_names_latex, sorted_mu_star, color='blue')
        ax2.set_xlabel(r"$\mu_{EE}^*$")
        ax2.set_ylabel('Paramètres')
        ax2.set_title(title + ' ' + cas)
        ax2.tick_params(axis='y', labelsize=9)
        fig2.subplots_adjust(left=0.4)
        fig2.tight_layout()
        if save:
            fig2.savefig(name2)
        plt.show()
        
    elementary_effect = morris_analyze._compute_elementary_effects(model_inputs = trajectories, model_outputs=out, trajectory_size=len(trajectories), delta=1/3)
    
    return results,elementary_effect


def funnel_graph(basePath, obj_names, obj_units, results_sensitivity, problem_AS, F, cas, nb_para_var,
                 save_figures=False, log_scale=False, split_graph = False):
    """
    Génère les graphiques type entonnoir


    Parameters
    ----------
    basePath : str
        chemin de sauvegarde
    obj_names : numpy.ndarray
        2D array 1ere ligne contient les noms
    obj_units : numpy.ndarray A 2D array where the first row contains the units for each objective.
    results_sensitivity
        The results from the sensitivity analysis.
    problem_AS ; dict
        dictionnaire du problème d'AS

    F : np array
        valeurs des critères
    cas : str
        string pour identifier le cas traîté
    nb_para_var : int
        nombre de paramètres
    save_figures : boolean
        
    log_scale : boolean
    
    Returns
    -------
    only plots
    """
    A = []
    for objective_index in range(F.shape[1]):
        A.append(get_sensitivity_plot_data(results_sensitivity, problem_AS['names'], objective_index=objective_index))

    # Labels
    xlabels = [f"Variation in {obj_units[0, i] if obj_units[0, i] != ' ' else '∅'}" for i in range(obj_units.shape[1])]
    titles = [f"Sensitivity {obj_names[0, i]} {cas}" for i in range(obj_names.shape[1])]

    xlims = [None] * obj_names.shape[1]
    plot_names = [basePath + '_Python\\save_plot_morris\\'+cas+'\\'f'sensitivity_obj_{obj_names[0, i]}_{cas}.pdf' for i in range(obj_names.shape[1])]

    # Plotting
    for k, a in enumerate(A):
        if split_graph==False and nb_para_var > 24 :
            split_graph = True
        if split_graph:
            split_indices = [(0, nb_para_var)] if not split_graph else [
                (0, nb_para_var // 2), 
                (nb_para_var // 2, nb_para_var)
            ]
    
            suffixes = ['_part1', '_part2']
            
            # si on split alors la boucle suivant tournera 2x
            # si sur true alors on a [0,11] suffix, [11,22]suffix2
            for (start, end), suffix in zip(split_indices, suffixes):
    
                plt.figure(figsize=(6.5, 8))
                plt.axvline(0, color='black', linewidth=1.0)
    
                for param_index in range(start, end):
                    plt.scatter(a[4][param_index], a[5][param_index], c='black', label='EE' if param_index == start else None)
                    
                
                plt.scatter(a[2][start:end], a[1][start:end], s=200, marker="h", c='red', label=r"$\mu_{EE}^*$")
                plt.scatter(a[3][start:end], a[1][start:end], s=150, marker="*", c='blue', label=r"$\sigma_{EE}$")
    
                for param_index in range(start, end):
                    plt.annotate(f"${round(a[2][param_index], ndigits=1)}$ {obj_units[0, k]}",
                                 (a[2][param_index] * 0.8, a[1][param_index] + 0.15),
                                 fontsize=10., color='red')
    
                param_names_latex = [f"${name}$" for name in a[0][start:end]]
                plt.yticks(a[1][start:end], param_names_latex, fontsize=14)
                #plt.ylim((a[1][start] - 0.5, a[1][end - 1] + 0.5))
                #plt.ylim((-0.5,end-start - 0.5))
                if log_scale:
                    plt.xscale('symlog')
    
                plt.xlabel(xlabels[k], fontsize=12)

    
                if suffix == '_part1':
                    xmin,xmax = plt.xlim()
                elif suffix == '_part2':
                    plt.xlim(xmin,xmax)

    
                plt.title(titles[k] + f" {suffix.replace('_', ' ')}\n\n", fontsize=14)
                plt.grid(linestyle="--", linewidth=0.1)
                plt.legend(fontsize=12., bbox_to_anchor=(0., 1.), ncol=3, loc='lower left', shadow=True)
                plt.tight_layout()
            
        else:
            plt.figure(figsize=(6.5, 8))
    
            # Line at 0
            plt.axvline(0, color='black', linewidth=1.0)
            n_trajectories = len(a[4][0])

            # Plot all EEi
            for param_index in range(nb_para_var):
                if param_index > 0:
                    plt.scatter(a[4][param_index], a[5][param_index], c='black')
                else:
                    plt.scatter(a[4][param_index], a[5][param_index], c='black', label='EE')
    
            # Scatter of mu and sigma
            plt.scatter(a[2], a[1], s=200, marker="h", c='red', label=r"$\mu_{EE}^*$")
            plt.scatter(a[3], a[1], s=150, marker="*", c='blue', label=r"$\sigma_{EE}$")
            
            # Annotations for each parameter
            for param_index in range(nb_para_var):
                plt.annotate(f"${round(a[2][param_index], ndigits=1)}$ {obj_units[0, k]}",
                             (a[2][param_index] * 0.8, a[1][param_index] + 0.15),
                             fontsize=10., color='red')
    
            # Convert parameter names to LaTeX format
            param_names_latex = [f"${name}$" for name in a[0]]
    
            # Names of parameters
            plt.yticks(a[1], param_names_latex, fontsize=14)
            plt.ylim((-0.5, nb_para_var - 0.5))
    
            # Log scale for x-axis if specified
            if log_scale:
                plt.xscale('symlog')
    
            # x label to use
            plt.xlabel(xlabels[k], fontsize=12)
    
            # x limits if specified
            if xlims[k] is not None:
                plt.xlim(xlims[k])
    
            plt.title(titles[k] + "\n\n", fontsize=14)
            plt.grid(linestyle="--", linewidth=0.2)
            plt.legend(fontsize=12., bbox_to_anchor=(0., 1.), ncol=3, loc='lower left', shadow=True)
            plt.tight_layout()
    
        if save_figures:
            plt.tight_layout()
            plt.savefig(plot_names[k])
        # plt.close()



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








