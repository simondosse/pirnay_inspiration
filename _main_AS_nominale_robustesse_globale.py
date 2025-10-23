# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 15:12:31 2025

@author: ZDO
Version de Python : 3.11.11 | packaged by conda-forge | (main, Dec  5 2024, 14:06:23) [MSC v.1942 64 bit (AMD64)]
"""

#########################
import os
os.chdir("L:\\ZDO\\1_AS_OPTIM\\"+"_Python")
import sys

current_folder = os.getcwd()
parent_folder = os.path.dirname(current_folder)
grandparent_folder = os.path.dirname(parent_folder)


basePath = os.path.join(parent_folder, '')  
csvPath = os.path.join(basePath, 'files_csv', '')  


print("grandparent_folder :", grandparent_folder)
print("basePath      :", basePath)
print("csvPath       :", csvPath)
#########################

sys.path.append(os.path.join(grandparent_folder,"_Python_commun"))
sys.path.append(os.path.join(parent_folder))

from valeur_from_excel import plage_valeur, middle_parameters
from morris_method import generate_trajectories,resultat, funnel_graph
from robustesse_incertitude import generate_pop, variation_p, recreate_plage, calculate_ext_tot_DEV
from _fonctions_generales import flatten_one_level, find_index_name, generate_map, generate_map_full, plot_table

import numpy as np
import pandas as pd
from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
import matplotlib.pyplot as plt



#%% INITIALISATION DU PROBLEME D'ANALYSE DE SENSIBILITE

"""
Ce script n'est utilisé que pour :
    - une analyse de sensiblité nominale p^NOM
    - une analyse de sensibilite de robustesse globale p^NOM+p^DEV
Ce script ne peut pas être utilisé pour une analyse de robustesse locale.
Pour l'analyse de robustesse locale il faut utiliser le script '_main_robustesse_locale.py'
"""

# On récupère les plages de paramètres depuis l'excel
# On choisit soit le CAS GENERAL, soit le FRET, soit le METRO
# /!\ pour le CAS_GENERAL certaines combinaisons de paramètre peut mener à un train instable sur une voie en alignement

robust = False
cas = "METRO"
sheet_excel = cas

# PLAGE DE VARIATIONS NOMINALES DES PARAMETRES : ext_tot_NOM
ext_tot_NOM, nb_para_var = plage_valeur(path_excel = basePath+"plage_valeur.xlsx",sheet_excel=sheet_excel)

# PLAGE DE VARIATIONS DEVAITIONS DES PARAMETRES : ext_tot_DEV
p = 0.05
ext_tot_DEV = calculate_ext_tot_DEV(ext_tot_NOM, p)

# On utilise ces plages pour crée les trajectoire de morris que l'on
# sauvegarde dans un fichier .csv

nb_lvl = 4 # Nombre de valeurs que peut prendre chaque paramètre


# On définit le problem initial où p = p^NOM

problem_AS_NOM = {
    'num_vars': nb_para_var,  # Nombre de paramètres
    'names': [r'd_w', r'd_b', r'm_c', r'm_b', r'm_e', 
              r'x_{cdg-c}', r'y_{cdg-c}', r'z_{cdg-c}',
              r'y_{prim}', r'k_{x-prim}', r'k_{y-prim}', r'k_{z-prim}', r'c_{z-prim}', 
              r'y_{sec-exc}', r'z_{sec-exc}', r'k_{x-sec-exc}', r'k_{y-sec-exc}', 
              r'k_{z-sec-exc}', r'c_{z-sec-exc}', r'c_{y-sec-pivot}', 
              r'x_{sec-roulis}', r'k_{x-sec-roulis}'],  # Noms des paramètres en LaTeX
    'bounds': [ext_tot_NOM[i] for i in range(len(ext_tot_NOM))]  # Plages
}
#problem_AS_NOM['names']=[name + r'^{NOM}' for name in problem_AS_NOM['names']]

# On définit le problem qui va prendre en compte p = p^NOM + p^DEV
# Le problème est par défition 2x plus grand en taille si chaque paramètre présente une déviation
problem_AS_robustesse = problem_AS_NOM.copy() # faut bien mettre une copy sinon ça pointe sur le même dictionnaire donc on finit pas modifier AS_NOM ça point sur même adresse
problem_AS_robustesse['names'] = problem_AS_NOM['names'] + [name + r'^{DEV}' for name in problem_AS_NOM['names']]
problem_AS_robustesse['num_vars'] = len(problem_AS_robustesse['names'])
problem_AS_robustesse['bounds'] = problem_AS_NOM['bounds'] + [ext_tot_DEV[i] for i in range(len(ext_tot_DEV))]

# On définit le problem que l'on va utiliser ainsi que le ext_tot
ext_tot = ext_tot_NOM

if robust == True:
    ext_tot = ext_tot_NOM + ext_tot_DEV # ça vient juste les concaténer
    problem_AS = problem_AS_robustesse
    cas = cas + '_' + 'robust_global'
else:
    problem_AS = problem_AS_NOM



nb_para_var = problem_AS['num_vars']
name_file_traj = csvPath+"trajectories_" + cas + ".csv"
#name_file_data = csvPath+"trajectories_data_" + cas + ".csv"
trajectories = generate_trajectories(name_file_traj, ext_tot, nb_para_var,nb_lvl,seed=22, problem = problem_AS)


# On génère une population avec 5% d'incertitude pour chaque paramètre
generate_pop(csvPath+"trains_incertitude.csv",p=5, nb_trains=10, nb_para_var=nb_para_var, ext_tot=ext_tot)

# On crée le .csv d'un véhicule classique dont les paramètres sont au milieu de chaque plage de valeurs
# on save ces paramètres dans un .csv
path_save =csvPath+"para_veh_middle_"+sheet_excel+".csv"
mid_para = middle_parameters(path_excel=basePath+"plage_valeur.xlsx", path_save=path_save, sheet_excel=sheet_excel)




#%% SIMULATION A FAIRE SUR VOCO ou SIMPACK
'''
Pour connaître la valeur des objectifs d'intérêt en fonction
du jeu de paramètres choisi pour chaque individu de la population
On obtient donc F(traj)
'''

#%% LECTURE DES RESULTATS DE LA SIMULATION et plots des mu / sigma
plt.close('all')

F = pd.read_csv(csvPath+'OUTvoco_'+cas+'.csv').to_numpy(dtype=float)
obj_names = pd.read_csv(csvPath+'obj_names_'+cas+'.csv').to_numpy(dtype=str)
obj_units = pd.read_csv(csvPath+'obj_units_'+cas+'.csv').to_numpy(dtype=str)

# On traite nos objectifs un par un
# On trace les sigma et mu* pour chaque objectif
# On trace les diagrammes en barres pour les mu* pour chaque paramètre pour chaque objectif

# --- Ces graphs ne sont pas les plus importants de l'analyse de sensibilité

for i in range(F.shape[1]):
    results, elem = resultat(problem_AS,
                             trajectories, 
                             F[:,i],
                             nb_lvl,
                             obj_names[0,i],
                             cas,
                             basePath=basePath)
    # les paramètres retournés par results et elem ne seront pas utilisés, on ne rien renvoyer en soit

# mais dcp si on fait ça on regarde objectif par objectif et c'est pas du multi objectif,
# on peut pas rendre le tout scalaire comme pour la partie analyse de sensibilité mhmh??
# => En fait pour l'AS on fait bien obj par obj, et ça va nous servir à selectionner les para à optimiser pour la suite


#%%  Pour tracer les graphs de Morris en entonnoir

# On trace ici les graphs les + importants pour l'AS

import copy
import Global_parameters_sensitivity as gps
# Les lignes 539 à 556 ont été changées car _check_group() n'existe plus dans la nouvelle version de SALib

plt.close('all')

function_values_Morris = []
for index in range(F.shape[1]):
    function_values_Morris.append(copy.deepcopy(F[:, index])) 


# OBJET DE SALib pour pouvoir tracer les graphs de Morris en entonnoir par la suite
results_sensitivity = gps.recompute_Morris_sensitivity(problem_AS['names'], nb_lvl,
                                                        trajectories,
                                                        function_values_Morris)

# On trace les FUNNEL_GRAPH
save_figures = True
log_scale = False
split_plot = False
funnel_graph(basePath, obj_names, obj_units, results_sensitivity, problem_AS, F, cas, nb_para_var,
             save_figures,
             log_scale,
             split_plot)


#%% SELECTION (pour le moment arbitraire 22 04) DES PARAM à OPTIMISER


# On sélectionne les paramètres que l'on considère influent
# on récupère les indices

# --- SELECTION DES PARAMETRES à OPTIMISER

# Sélection des paramètres à optim dans parmi problem_AS['names']
para_to_optim = ['y_{cdg-c}','z_{cdg-c}','m_c','k_{y-sec-exc}']
para_units = ['m','m','kg','N/m']

# associe les noms des para à optimiser à leurs unités
para_pairs = list(zip(para_to_optim, para_units))
# on récupère les indices des para_to_optim dans problem_AS['names']
# on remet dans l'ordre les para_units aussi
# on fait ça pour que ce soir dans le même ordre d'apparition que problem_AS['names'] pour rester cohérent
sorted_pairs = sorted(para_pairs, key=lambda pair: problem_AS['names'].index(pair[0]))

para_to_optim_sorted, para_units_sorted = map(np.array, zip(*sorted_pairs))

ind_to_optim = find_index_name(problem_AS['names'],para_to_optim)


# --- SELECTION DES OBJECTIFS à OPTIMISER
'''
'Qmin_static', 'Ymax_static', 'dQQ0_static', 'XFactor_static',
        'YQmax_static', 'Vc_dyn', 'SumYmax_dyn', 'Ymax_dyn'
'''
obj_names_flat = obj_names.flatten()
selected_obj= ['dQQ0_static', 'YQmax_static']

ind_selected_obj = [int(np.where(obj_names_flat == name)[0][0])+1 for name in selected_obj] # on rajoute +1 car on va renvoyer ça à MATLAB où les tableaux commencent à 1 et non 0

# Sauvegarde des indices des objectifs à sauvegarder
# seuls ces obj seront calculés sur MATLAB
pathSave_ind_selected_obj = csvPath + 'ind_selected_obj.csv'
np.savetxt(pathSave_ind_selected_obj, ind_selected_obj, delimiter=',')

n = 4 # nombre de valeur par plage par paramètre
# (même nombre pour l'ensemble des paramètres, à voir si on fait un maillage 
# + fin pour certains paramètres ?)

# La fonction suivante n'est pas utile ici, elle permettait de créer une map representant un bon nombre de combinaisons possibles entre les paramètres à optim
# mais cette approche d'évaluer l'ensemble de la map F(map) est bcp trop chronophage -> Algo NSGA2 bien plus rapide, il ne teste pas toutes les combi évidemment
map_full = generate_map_full(problem_AS,
                             para_to_optim,
                             ext_tot,
                             n,
                             nb_para_var,
                             mid_para)

#%% OPTIMISATION AVEC L'ALGORITHME NSGA2

from optimisation_train import Problem_optim_train
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultMultiObjectiveTermination
import matlab.engine

# Définition de la borne inférieure xl et supérieure xu des paramètres sélectionnés
xl = []
for index in sorted(ind_to_optim):
    xl.append(problem_AS['bounds'][index][0])
    
xu = []
for index in sorted(ind_to_optim):
    xu.append(problem_AS['bounds'][index][1])

# On lance un environnement MATLAB pour pouvoir appeler des fonctions MATLAB
# /!\ bien penser à fermer le eng avant d'en relancer
eng = matlab.engine.start_matlab()


"""
L'idée centrale en optimisation multi-objectifs est de trouver un ensemble de solutions non dominées,
c'est-à-dire qu'on ne peut pas améliorer un objectif sans détériorer au moins un autre.
"""

"""
/!\ le n_obj doit déjà être réduit ici si on fait pas une optim multi-objectif
sur tous les critères de l'AS, on va par exemple seulement en choisir 2
"""


n_obj = 2 # nombre d'objectifs à optim en même temps (F)
n_ieq_constr = 2 # nombre de contraintes à respecter (G)

# On définit le problème d'optimisation avec notamment la fonction _evaluate() qui sera appelée plusieurs fois par la suite
problem_optim = Problem_optim_train(n_var = len(para_to_optim_sorted), n_obj = n_obj, 
                                    n_ieq_constr = n_ieq_constr, # nombre de contrainted par inéquation
                                    xl=xl, xu=xu, # bornes inf et sup de tous nos paramètres variables
                                    ind_to_optim=sorted(ind_to_optim), # indices des para qu'on va optim parmis les 22
                                    mid_para=mid_para, # valeur de l'ensemble des paramètres au centre de leur plage, on va renvoyer tous ces para à chaque fois avec ceux sélectionnés qui auront changer
                                    eng = eng, # engine matlab pour run les simu VOCO (s'active en arrière plan, si erreur avant fin d'optim, bien penser à le fermer avec eng.quit())
                                    basePath=basePath,
                                    grandparent_folder=grandparent_folder)

"""
/!\
nombre d'appels à la fonction self._evaluate() = pop_size + (n_gen*n_offsprings)
"""

# Définition de l'algorithme d'optimisation
algorithm_NSGAII=NSGA2( # NSGA hérite de GeneticALgorithm qui hérite de Algorithm
                pop_size=20, #taille de l'échantillon initial
                n_offsprings=10, #taille de chaque échantillon d'enfants créés à partir de la pop actuelle par le biais des opérateurs de croisement et mutation
                                    # un n_offsprings + élevé permet une exploration plus large de l'espace mais prend + de temps
                                    # dans GeneticAlgorithm
                                    # if self.n_offsprings is None:
                                    #      self.n_offsprings = pop_size
                sampling=FloatRandomSampling(), #Méthode d'échantillonage pour la population initiale : échantillonage aléatoire choisi, c'est comme ça par défaut si on renseigne rien 
                crossover=SBX(prob=0.9,eta=20), #Lois de probabilité définissant les opérateurs de croisement (par défait eta=15, prob = 0.9)
                mutation=PolynomialMutation(eta=20), #Lois de probabilité définissant les opérateurs de mutation
                eliminate_duplicates=True, # ce n'est pas un attribut de NSGA2 mais bien de la class dont il hérite GeneticAlgorithm, probablement gérer par le **kwargs
                termination=DefaultMultiObjectiveTermination( # une seule des conditions suivantes entraîne l'arrêt de l'algo
                                xtol=0.0005, # si les changements des paramètres entres gen sont inf à xtol alors on s'arrête. VALEUR de variation MOYENNE ????? ou par PARA ???
                                cvtol=1e-8, # par rapport aux contraintes ?
                                ftol=0.005, # si les changements dans les valeurs des obj entre 2 gen est < a ftol alors on s'arrête
                                n_max_gen=10) # nbr max de gen
                )



"""
Si on veut partir d'un jeu d'une population initiale donnée :
    from pymoo.core.population import Population
    pop = Population.new("X", [[0.1, 0.5], [0.3, 0.7], [0.9, 0.2]])  # Exemple pour 2 variables
    algorithm = NSGA2(pop_size=10, sampling=pop)
    
Sinon par défaut la population initiale est aléatoire suivant le sampling voulu
"""

# Lancement de l'algorithme d'optimisation
# minimize renvoie un :class:`~pymoo.core.result.Result`
res = minimize(problem_optim, # herite forcement de la classe Problem et doit présenter une fonction _evaluate() bien définie
                              # dans notre cas on évalue F(X) grâce aux simulations sur VOCO 
               algorithm_NSGAII, #objet algo optim, il définit le type d'algo d'optim utilisé
               ("n_gen", 5), # terminaison : critère d'arrêt pour l'algo, ici on fait n génération et on s'arrête
               verbose=True, # pour afficher ou non les info du processus d'optim, utile pour debug
               seed=1) # permet de reproduire les résultats en fixant la séquence aléatoire
                        # save_history(optionnel) sur True l'historique de générations sera enregistré pour un post traitement
# On ferme la machine MATLAB
eng.quit()

"""
Le nombre d'individus X présent sur le front de pareto peut varier si on lance plusieurs fois ce même code.
Si certains individus sont trop proches (en terme de valeur d'obj) alors certains peuvent être supprimés pour assurer une "diversité génétique"
                                       
len(res.F)<= len(pop_size) 
                                         
Il y a plusieurs front de Pareto à l'issu de l'optim :
- certaines sont non dominées entre elles : elles forment le front de Pareto n°1
- d'autres sont dominées seulement par celles du front 1 : elles forment le front n°2
- d'autres sont dominées par des solutions des fronts 1 et 2 : front n°3...... et ainsi d e suite

On peut récuper les autres front comme ça :

    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    F = res.F
    nds = NonDominatedSorting()
    fronts = nds.do(F, n_stop_if_ranked=3)  # Donne les 3 premiers fronts (liste de listes d'indices)


nous on va afficher uniquement le 1er front évidemment, c'est le plus optimal!
"""
#%% AFFICHAGE DES RESULTATS DE L'OPTIMISATION

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# tous les trains simulés
# all_X = []
# for gen in res.history:
#     all_X.append(gen.pop.get("X"))
# all_X = np.vstack(all_X)

# # tous les coûts évalués
# all_F = []
# for gen in res.history:
#     F_gen = gen.pop.get("F")
#     all_F.append(F_gen)
    
# all_F = np.vstack(all_F)

plt.close('all')

F = res.F
G = res.G
X = res.X

# F est de shape (N, 2)
# Colonne 0 = objectif 1, Colonne 1 = objectif 2
x = F[:, 0]
y = F[:, 1]

nds = NonDominatedSorting()
fronts = nds.do(F, n_stop_if_ranked=3)  # Donne les 3 premiers fronts (liste de listes d'indices)

# ------ PLOT FRONT DE PARETO (pour 2 obj)
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(x, y, color='red')
ax.set_xlabel(f'Objectif 1 - {selected_obj[0]}')
ax.set_ylabel(f'Objectif 2 - {selected_obj[1]}')
ax.set_title(f"Front de Pareto - Nombre objectifs : {n_obj}")
ax.legend()
ax.grid(True)

for i in range(len(x)):
    ax.annotate(fr"$X_{{{i+1}}}$", (x[i], y[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    """
    (x[i], y[i]) c’est la position du point sur le graphique

    textcoords="offset points" : pour dire comment on va traiter les coord xytext=(5,5), ici en points typo, pas en unité du graphique

    xytext=(5, 5) : on décale le texte de 5 points en X et 5 points en Y par rapport au point x[i] y[i]
    """
# ------ PLOT DU TABLEAU D'INDIVIDUS

fig_table,ax_table = plt.subplots(figsize=(6,8))

cell_text = []
for i in range(len(X)):
    ligne = [fr"$X_{{{i+1}}}$"]
    ligne += [f"{val:.3e}" for val in X[i]]  # format scientifique
    cell_text.append(ligne)
    
column_labels = [""] + [
    fr"${name.strip()}$ [{unit}]" for name, unit in zip(para_to_optim_sorted, para_units_sorted)
]

plot_table(ax_table, cell_text, column_labels, title='Détails des individus du front de Pareto')
fig_table.tight_layout()     
  
fig.savefig(os.path.join(basePath,"_Python","save_plot_optim",f"pareto_front_obj_{n_obj}.pdf"))         
fig_table.savefig(os.path.join(basePath,"_Python","save_plot_optim",f"pareto_front_table_obj_{n_obj}.pdf"))   


# On renvoie le train optimal choisi parmis les solutions optimales

train_opti = mid_para

for idx, val in zip(ind_to_optim, X):
    train_opti[idx] = round(val, 2)
    
np.savetxt(os.path.join(csvPath,'train_opti'), train_opti, delimiter=',')
























