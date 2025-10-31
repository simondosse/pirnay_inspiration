
#%%
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
from input import ModelParameters
import ROM
import plotter
import NACA
from SALib.sample import saltelli
from SALib.analyze import sobol
from data_manager import save_modal_data

from _functions_AS_optim import map_to_physical

#%% ------- SOBOL_______________________________________________________________________________

coeff_low, coeff_high = 0.6, 1.4

para_interval = np.array([
    [0.0, 1.0],                             # u x_ea/c [0.15,0.5]
    [0.0, 1.0],                             # v facteur d'écart du CG à 0.8c
    [coeff_low * 366, coeff_high * 366],
    [coeff_low * 78, coeff_high * 78]
])
'''
On pourrait reparamétrer explicitement en (x_ea/c, Δ_cg) avec Δ_cg = x_cg - x_ea
'''

nb_para = para_interval.shape[0]

problem_uv = {
    'num_vars': nb_para,  # Nombre de paramètres
    'names': [r'u_ea',r'v_cg',r'EIx',r'GJ'],  # Noms des paramètres en LaTeX
    'bounds': para_interval # Plages
}

np.random.seed(3) # we fix random to be able to repeat same results
N = 32  # base sample size
X_uv = saltelli.sample(problem_uv, N, calc_second_order=False) # calc_second_order=False divides /2 the number of evaluations
X = map_to_physical(X_uv)

# default model to be updated
s, c = 2.0, 0.2
m = 2.4
eta_w = 0.005
eta_alpha = 0.005
model = ModelParameters(s, c, x_ea=0.05, x_cg=0.10, m=m, EIx=400, GJ=70, eta_w=eta_w, eta_alpha=eta_alpha, model_aero='Theodorsen')






#%% F(X)_________________________________________________________________________________________________________

nb_obj = 2
F = np.zeros((len(X),nb_obj))
case_status = np.empty(len(X), dtype=object) # on dit que c'est un np qui peut contenir n'importe quoi, il contiendra des str

print('Evaluation of F(X) :') 
for i in range(len(X)):        
    model.airfoil.x_ea = X[i][0]*c # *c because we are dealing with adimensionnal parameter
    model.airfoil.x_cg = X[i][1]*c # peut faire en sorte que quand on appelle model.x_cg ça appelle en fait model.airfoil.x_cg ? moue c'est mieux de laisser comme ça : on comprend + ce que l'on fait
    model.EIx = X[i][2]
    model.GJ = X[i][3]

    f, damping, *_ = ROM.ModalParamDyn(model)
    Uc,slope_retained,case = ROM.damping_crossing_slope(U = model.U, damping = damping[:,0], return_status=True)

    F[i][0]=Uc
    F[i][1]=slope_retained
    case_status[i]=case
    print(f'F[{i+1}] / F[{len(X)}]')

F_status = np.hstack((F, case_status.reshape(-1, 1)))
np.savez('data/F_sobol', X_uv=X_uv, X_phys=X, F=F, F_full = F_status)







#%%__________________RESULTS______________________________________________________________

import pandas as pd

def summarize_Si(Si, names):
    return pd.DataFrame({
        'S1': Si['S1'],             # indice de sobol d'ordre 1 (effet direct de chaque paramètre pris seul)
        'S1_conf': Si['S1_conf'],   # intervalle de confiance associé à S1
        'ST': Si['ST'],             # indices totaux (effet gloabal d'un paramètre, incluant toutes ses interactions ave d'autres)
        'ST_conf': Si['ST_conf']    # intervall de confiance associé à ST
    }, index=names)

def barplot_indices(df, title):
    """
    Trace un barplot des indices de Sobol (S1, ST) avec leurs intervalles de confiance.
    """
    x = np.arange(len(df))  # positions des barres
    width = 0.35            # largeur des barres côte à côte

    fig, ax = plt.subplots(figsize=(8, 4))

    # Barres S1 (premier ordre)
    ax.bar(x - width/2, df['S1'], width, yerr=df['S1_conf'], 
           label='S1 (effet direct)', capsize=5, alpha=0.8)

    # Barres ST (total)
    ax.bar(x + width/2, df['ST'], width, yerr=df['ST_conf'], 
           label='ST (effet total)', capsize=5, alpha=0.8)

    # Mise en forme
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=0)
    ax.set_ylabel('Indice de Sobol')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# Si = sobol.analyze(problem_uv, F, calc_second_order=False, print_to_console=False)
# les indices de Sobol sont calculés objectif par objectif ?
data = np.load('data/F_sobol.npz')
F = data['F']
Si_Uc    = sobol.analyze(problem_uv, F[:, 0], calc_second_order=False, print_to_console=False)
Si_slope = sobol.analyze(problem_uv, F[:, 1], calc_second_order=False, print_to_console=False)
'''
sobol.analyze renvoie des dicos dont les keys sont les indices
'''

names = [r'$u$', r'$v$', r'$EI_x$', r'$GJ$']
Si_Uc_df    = summarize_Si(Si_Uc,    names)
Si_slope_df = summarize_Si(Si_slope, names)

barplot_indices(Si_Uc_df, 'Sobol — Uc')
barplot_indices(Si_slope_df, 'Sobol — pente')


# %%
