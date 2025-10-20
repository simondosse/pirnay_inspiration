#%%
''' Import the necessary libraries '''
import os
import numpy as np
import matplotlib.pyplot as plt
from input import ModelParameters
import ROM
import plotter
import NACA
from data_manager import save_modal_data, _load_npz


# plt.close('all')
# plt.ion()

# modifie tous les parametres de matplotlib
# --- Configuration globale Matplotlib ---
# plt.rcParams.update({
#     'figure.figsize': [7.17, 4.34],       # Taille des figures (format article)
#     # 'text.usetex': True,                  # Active le rendu LaTeX
#     # 'text.latex.preamble': r'\usepackage{mathptmx}',  # Police Times pour le texte et les maths
#     'font.family': 'serif',               # Police avec empattements
#     'font.size': 11,                      # Taille globale du texte
#     'axes.labelsize': 11,                 # Taille des labels d'axes
# })


#%% Set and run models, save data

''' Structural parameters '''
s = 2 #half span
c = 0.2 #chord
x_ea = c/3 # elastic axis location from leading edge
x_cg = 0.379*c # center of gravity location from leading edge
m = 2.4 # mass per unit span

res = NACA.inertia_mass_naca0015(c=c, mu=m, N=4000, span=s)
# I_alpha = 5.6e-3 # mass moment of inertia per unit span
I_alpha=res.Jz_mass+m*abs(x_cg-x_ea)**2 # parallel axis theorem to get torsional inertia about elastic axis
EIx = 366 # bending stiffness
GJ = 78 # torsional stiffness
eta_w = 0.005 # structural damping ratio in bending
eta_alpha = 0.005 # structural damping ratio in torsion, damping ratio are arbitrary choosen here

''' Wingtip parameters '''

wingtip_mass_study = False
if wingtip_mass_study:
    Mt = 362e-3   # mass of the tip body
    I_alpha_t = 6.11e-4 # mass moment of inertia of the tip body
                    # I_alpha_t must depends on x_t right ??
    x_t = 0.007  # location of the tip body from leading edge (from the elastic axis isnt it ?)
else:
    Mt = None
    I_alpha_t = None
    x_t = None

''' Set, run and save models '''

#Theodorsen model
model_theod = ModelParameters(s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t, 'Theodorsen')
f, damping, _ = ROM.ModalParamDyn(model_theod)

model_struc = ModelParameters(s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t)
eig_strucS2 = ROM.ModalParamAtRest(model_struc)
model_struc.update(s=1.5)
eig_strucS15 = ROM.ModalParamAtRest(model_struc)


# save_modal_data(f = f, damping = damping, model_params=model_theod,out_dir='data', filename='model_params_TheodorsenS2.npz')

# #QuasiSteady model
# model_qs = ModelParameters(s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t, 'QuasiSteady')
# f, damping, _ = ROM.ModalParamDyn(model_qs)
# save_modal_data(f = f, damping = damping, model_params=model_qs,out_dir='data', filename='model_params_QuasiSteadyS2.npz')

''' Parametric studies '''
# x_t_new = np.linspace(0, 0.02, 10)

# for x_t_val in x_t_new:
#     model.update(x_t=x_t_val) # we can update several parameters at once because it's a kwargs 
#                              # while the key is the name of the parameter in the class ModelParameters
#     print(f"Updated model with mass: {model.x_t}")
#     run_model(model)
"""
this will create a lot of plots, one for each mass value, we should put the results in a dataset and plot them all together
on a another script that only deals with plotting
databse should be .csv ?
"""

#%%------------------------------------------------
hop= False
if hop:
    plotter.plot_modal_data_single(npz_path='data/model_params_Theodorsen.npz')

    #%%------------------------------------------------


    data = _load_npz('data/model_params_Theodorsen.npz')
    fig, ax = plt.subplots()
    ax.plot(data['U'], data['damping'][:, 0],label='0')
    ax.plot(data['U'], data['damping'][:, 1],label = '1')
    ax.legend()


    data = _load_npz('data/model_params_Theodorsen.npz')
    fig, ax = plt.subplots()
    ax.plot(data['U'], data['f'][:, 0],label='0')
    ax.plot(data['U'], data['f'][:, 1],label = '1')
    ax.legend()

    #%%--------------------------------------------------------

    plotter.plot_modal_data_two(npz_path_a='data/model_params_Theodorsen.npz',
                        npz_path_b='data/model_params_QuasiSteady.npz')
    plotter.plot_modal_data_two(npz_path_a='data/model_params_TheodorsenS2.npz',
                        npz_path_b='data/model_params_QuasiSteadyS2.npz')
    # plot_params_table('data/model_params_Theodorsen.npz')

#%%----------------------------- NACA 0012 airfoil shape plotter




NACA.plot_naca00xx_section_with_cg(t_c=0.15, c=c, N=4000, xcg=x_cg, fill=True, annotate=True)


# %%
