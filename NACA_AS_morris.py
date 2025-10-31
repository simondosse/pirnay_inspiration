#%%
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
from input import ModelParameters
import ROM
import plotter
import NACA
from morris_method import generate_trajectories,resultat, funnel_graph
from SALib.sample import saltelli
from SALib.analyze import sobol
from data_manager import save_modal_data

'''
This script is used to perform sensitivity analysis (Method or Sobol)
for NACA section's parameters
'''




#%% MORRIS_________________________________________________________________________________________________________________
coeff_low, coeff_high = 0.6, 1.4

'''
we might evaluate de x_ea and x_cg respect to the aerodynamical center ????
'''
para_interval = np.array([
    [0.15, 0.4],                            # x_ea/c
    [0.4, 0.8],                             # x_cg/c
    [coeff_low * 366, coeff_high * 366],    # EIx
    [coeff_low * 78, coeff_high * 78]       # GJ
])
nb_para = para_interval.shape[0]

problem= {
    'num_vars': nb_para,  # Nombre de paramètres
    'names': [r'x_{ea}/c',r'x_{cg}/c',r'EIx',r'GJ'],  # Noms des paramètres en LaTeX
    'bounds': para_interval # Plages
}

# Sampling
X = generate_trajectories(problem = problem, nb_traj_opti=8) # watch out the seed setup (seed= 1, we fixe the randomness)






#%%_____________________F(X)____________________________________________
s, c = 2.0, 0.2
m = 2.4
eta_w = 0.005
eta_alpha = 0.005
model = ModelParameters(s, c, x_ea=0.05, x_cg=0.10, m=m, EIx=400, GJ=70, eta_w=eta_w, eta_alpha=eta_alpha, model_aero='Theodorsen')


nb_obj = 2
F = np.zeros((len(X),nb_obj))
case_status = np.empty(len(X), dtype=object) # on dit que c'est un np qui peut contenir n'importe quoi, il contiendra des str
damping_box = np.zeros((len(X),80,2))
f_box = np.zeros((len(X),80,2))

print('Evalution of F(X) :') 
for i in range(len(X)):

    model.airfoil.x_ea = X[i][0]*c # *c because we are dealing with adimensionnal parameter
    model.airfoil.x_cg = X[i][1]*c # peut faire en sorte que quand on appelle model.x_cg ça appelle en fait model.airfoil.x_cg ? moue c'est mieux de laisser comme ça : on comprend + ce que l'on fait
    model.EIx = X[i][2]
    model.GJ = X[i][3]

    f, damping, f_modes_U , w_modes_U, a_modes_U = ROM.ModalParamDyn(model)
    damping_box[i,:,0] = damping[:,0]
    damping_box[i,:,1] = damping[:,1]
    f_box[i,:,0] = f[:,0]
    f_box[i,:,1] = f[:,1]

    Uc,slope_retained,case = ROM.damping_crossing_slope(U = model.U, damping = damping[:,0], return_status=True)

    F[i][0]=Uc
    F[i][1]=slope_retained
    case_status[i]=case
    print(f'F[{i+1}] / F[{len(X)}]')

F_status = np.hstack((F, case_status.reshape(-1, 1)))
np.savez('data/F_morris',X=X,F=F,F_status=F_status)








#%% Trying a set of parameter____________________________________________________________
'''
j = 0
x_ea = X[j][0]*c
x_cg = X[j][1]*c
EIx = X[j][2]
GJ = X[j][3]
res = NACA.inertia_mass_naca0015(c=c, mu=m, N=4000, span=s, xcg_known=x_cg)
I_alpha=res.Jz_mass+m*abs(x_cg-x_ea)**2
model = ModelParameters(s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, None, None, None,'Theodorsen')
f, damping , f_modes_U , w_modes_U, a_modes_U = ROM.ModalParamDyn(model,normalize=None) # normalize = None or 'per_field' or 'per_mode'
Uc,slope_retained = ROM.damping_crossing_slope(U = model.U, damping = damping[:,0])

save_modal_data(f = f, damping = damping, model_params=model,out_dir='data', filename='model_test_Morris.npz')
plotter.plot_modal_data_single(npz_path='data/model_test_Morris.npz')
'''




#%% Results____________________________________________________________________________________________________________________________
F = np.load('data/F_morris.npz')['F']
obj_names = np.array([r'$U_c$',r'$\frac{d\zeta}{dU}$'])
obj_units = np.array([r'$m/s$',r'_'])
'''
- We process one objective at a time
- sigma and mu* for each objective
- bar diagram
'''
# On traite nos objectifs un par un
# On trace les sigma et mu* pour chaque objectif
# On trace les diagrammes en barres pour les mu* pour chaque paramètre pour chaque objectif
for i in range(F.shape[1]):
    results, elem = resultat(problem,
                             X, 
                             F[:,i],
                             nb_lvl=4,
                             title = obj_names[i],
                             cas = 's fixed')

import copy
import Global_parameters_sensitivity as gps
# Les lignes 539 à 556 ont été changées car _check_group() n'existe plus dans la nouvelle version de SALib

plt.close('all')

function_values_Morris = []
for index in range(F.shape[1]):
    function_values_Morris.append(copy.deepcopy(F[:, index])) 


# OBJET DE SALib pour pouvoir tracer les graphs de Morris en entonnoir par la suite
results_sensitivity = gps.recompute_Morris_sensitivity(problem['names'], n_levels=4,
                                                        samples_array=X,
                                                        function_values_list=function_values_Morris)

# On trace les FUNNEL_GRAPH
save_figures = True
log_scale = False
split_plot = False
funnel_graph(obj_names, obj_units, results_sensitivity, problem, F, cas='s=2', nb_para_var = nb_para,)


# %%
