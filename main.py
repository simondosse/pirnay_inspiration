#%%
''' Import the necessary libraries '''
import os
import numpy as np
import matplotlib.pyplot as plt
from input import ModelParameters
import ROM
import plotter
import NACA
from _functions_AS_optim import *

from data_manager import save_modal_data, _load_npz
from scipy.linalg import eigh

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

#%%_________________________________________________
''' Set, run and save models '''

#Theodorsen model
model_theod = ModelParameters(s, c, x_ea, x_cg, m, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t, model_aero='Theodorsen')
f, damping, _ = ROM.ModalParamDyn(model_theod)

model_struc = ModelParameters(s, c, x_ea, x_cg, m, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t)
eig_strucS2 = ROM.ModalParamAtRest(model_struc)
model_struc.update(s=1.5)
eig_strucS15 = ROM.ModalParamAtRest(model_struc)




#%%___________________________________TEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEST

model = ModelParameters(s, c, x_ea, x_cg, m, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t,'Theodorsen')
f, damping, _ , _ , _ = ROM.ModalParamDyn(model)
save_modal_data(f = f, damping = damping, model_params=model,out_dir='data', filename='model_test.npz')
plotter.plot_modal_data_single(npz_path='data/model_test.npz')





#%%________________________________________
# #QuasiSteady model
model_qs = ModelParameters(s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t, 'QuasiSteady')
f, damping, _ = ROM.ModalParamDyn(model_qs)
save_modal_data(f = f, damping = damping, model_params=model_qs,out_dir='data', filename='model_params_QuasiSteadyS2.npz')

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

#%%--------------------------------------------------------

    plotter.plot_modal_data_two(npz_path_a='data/model_params_Theodorsen.npz',
                        npz_path_b='data/model_params_QuasiSteady.npz')
    plotter.plot_modal_data_two(npz_path_a='data/model_params_TheodorsenS2.npz',
                        npz_path_b='data/model_params_QuasiSteadyS2.npz')
    # plot_params_table('data/model_params_Theodorsen.npz')

#%%----------------------------- NACA 0012 airfoil shape plotter

NACA.plot_naca00xx_section_with_cg(t_c=0.15, c=c, N=4000, xcg=x_cg, fill=True, annotate=True)


# %%______________________________________________________________________________
# plot mode shape

# Modèle structurel seul (sans aéro)
s=2
model_struc = ModelParameters(s, c, x_ea, x_cg, m, EIx, GJ, eta_w, eta_alpha)
f0, zeta0, eig0, V0, w_modes, alpha_modes = ROM.ModalParamAtRest(model_struc, normalize='per_mode') # normalize = 'per_field' or 'per_mode'
plotter.plot_mode_shapes_grid(y=model_struc.y, freqs_hz=f0, W=w_modes, ALPHA=alpha_modes, normalize=False, suptitle='Structural mode shape contribution')






# %%_____________________________________________________________________________
s=1.5


model_struc = ModelParameters(s, c, x_ea, x_cg, m, EIx, GJ, eta_w, eta_alpha,model_aero='Theodorsen')
# model_struc.Umax = 100                          # Maximum velocity of the IAT wind tunnel
# model_struc.steps = 200                        # Number of velocity steps
# model_struc.U = np.linspace(0.1, model_struc.Umax, model_struc.steps)

f, damping , f_modes_U , w_modes_U, a_modes_U = ROM.ModalParamDyn(model_struc,normalize=None) # normalize = None or 'per_field' or 'per_mode'

save_modal_data(f = f, damping = damping, model_params=model_struc,out_dir='data', filename='model_test.npz')
plotter.plot_modal_data_single(npz_path='data/model_test.npz')



# mode shape contributions plot
plotter.plot_mode_shapes_over_U_grid(y=model_struc.y,U=model_struc.U,
                                    WU=w_modes_U, # shape (nU, n_modes, Ny), déjà normalisé si souhaité
                                    ALPHAU=a_modes_U, # idem
                                    f_modes_U=f_modes_U,
                                    mode_indices=[0,1,2,3,4], # modes #2 et #3 (1- ou 0-based accepté)
                                    n_samples=10,
                                    sharey=True, 
                                    suptitle='Aeroelastic mode shapes across U')




#%%___________________________________________RESOLUTION TEMPORELLE____________________________________

algo_name = 'DE'
data = np.load(f'data/res_{algo_name}.npz')
X_opt = map_to_physical(data['resX'])
x_ea = X_opt[0]*c
# x_ea = 0.075
x_cg = X_opt[1]*c
# x_cg = 0.075
EIx = X_opt[2]
GJ = X_opt[3]
model_opt = ModelParameters(s=s, c=c, x_ea=x_ea, x_cg=x_cg, m=m, EIx=EIx, GJ=GJ, eta_w=eta_w, eta_alpha=eta_alpha,model_aero='Theodorsen')
model_opt.Umax = 25
model_opt.Nw=3
model_opt.airfoil.plot_naca00xx_section()

f0, zeta0, eig0, V0, w_modes, alpha_modes = ROM.ModalParamAtRest(model_opt, normalize='pfer_mode') # normalize = 'per_field' or 'per_mode'
plotter.plot_mode_shapes_grid(y=model_opt.y, freqs_hz=f0, W=w_modes, ALPHA=alpha_modes, sharey=True, suptitle='Structural mode shape contribution - U=0')

f, damping, f_modes_U, w_modes_U, alpha_modes_U = ROM.ModalParamDyn(model_opt,tracked_idx=(0,1,2,3))
save_modal_data(f = f, damping = damping, model_params=model_opt,out_dir='data', filename='model_optim.npz')
plotter.plot_modal_data_single(npz_path='data/model_optim.npz' )

# %matplotlib widget
t0 = 0
tf = 10
dt = 0.001
t = np.arange(t0, tf+dt, dt)

X0 = ROM.build_state_q_from_real(
    par=model_opt,
    w_tip=0.01,        # m
    alpha_tip=0.01,     # rad
    wdot_tip=0.0,
    alphadot_tip=0.0
) # même si on veut simuler une réponse temporelle avec un U!=0 faut mettre un petit w0 ou aplha0 sinon tous les efforts symétriques s'annulent

'''
thanks to range kutta we get the temporal solutions of a initial state X0 and a freestream speed U
then we plot w(y,t) alpha(y,t)
then we plot w(y=s,t) alpha(y=s,t) + FFT
'''

U=0
t,X,A = ROM.integrate_state_rk(par = model_opt, U = U , t=t, x0 = X0, rk_order=4)

ROM.plot_w_alpha_fields(par = model_opt, t=t, X=X,U=U,times_to_plot = np.linspace(t[0],t[-1],10))
ROM.plot_tip_time_and_fft(par = model_opt, t=t,X=X, U=U, detrend=True)

plotter.plot_mode_shapes_over_U_grid(y=model_opt.y,U=model_opt.U,
                                    WU=w_modes_U, # shape (nU, n_modes, Ny), déjà normalisé si souhaité
                                    ALPHAU=alpha_modes_U, # idem
                                    f_modes_U=f_modes_U,
                                    mode_indices=[0,1,2,3,4,5], # modes #2 et #3 (1- ou 0-based accepté)
                                    n_samples=10,
                                    sharey=True, 
                                    suptitle='Aeroelastic mode shapes across U')





#%%_______________________________________Optimal section_______________________________________________









# %%
