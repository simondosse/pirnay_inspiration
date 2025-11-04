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

# #Theodorsen model
# model_theod = ModelParameters(s, c, x_ea, x_cg, m, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t, model_aero='Theodorsen')
# f, damping, _ = ROM.ModalParamDyn(model_theod)

# model_struc = ModelParameters(s, c, x_ea, x_cg, m, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t)
# eig_strucS2 = ROM.ModalParamAtRest(model_struc)
# model_struc.update(s=1.5)
# eig_strucS15 = ROM.ModalParamAtRest(model_struc)





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

f0, zeta0, eigvals0, eigvecs0, w_modes, alpha_modes, energy_dict = ROM.ModalParamAtRest(model_opt) # normalize = 'per_field' or 'per_mode'
Vq = eigvecs0[:model_opt.Nq, :]

plotter.plot_mode_shapes_grid(y=model_opt.y, freqs_hz=f0, W=energy_dict['T_ew'], ALPHA=energy_dict['T_ea'], sharey=True, suptitle='Structural mode shape contribution - U=0')
plotter.plot_vi_grid(Vq=Vq, Nw=model_opt.Nw, Nalpha=model_opt.Nalpha, freqs_hz=f0, kind='abs', normalize='l2', sharey=True, suptitle='Modal coefficients per mode')



# for i in range(model_opt.Nq):
#     plotter.plot_vi(vi = Vq[:,i], Nw = model_opt.Nw, Nalpha = model_opt.Nalpha)


#%%
f, damping, f_modes_U, w_modes_U, alpha_modes_U, energy_dict_U = ROM.ModalParamDyn(model_opt,tracked_idx=(0,1,2,3))
save_modal_data(f = f, damping = damping, model_params=model_opt,out_dir='data', filename='model_optim.npz')
plotter.plot_modal_data_single(npz_path='data/model_optim.npz' )
#%%
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
                                    # WU=w_modes_U, # shape (nU, n_modes, Ny), déjà normalisé si souhaité
                                    WU = energy_dict_U['T_ew_U'],
                                    # ALPHAU=alpha_modes_U, # idem
                                    ALPHAU = energy_dict_U['T_ea_U'],
                                    f_modes_U=f_modes_U,
                                    mode_indices=[0,1,2,3,4,5], # modes #2 et #3 (1- ou 0-based accepté)
                                    n_samples=10,
                                    sharey=True, 
                                    suptitle='Aeroelastic mode shapes across U')

#%%______________plot animation

fig,ani = plotter.animate_beam(par=model_opt, t=t, X=X, U=U, n_stations=12, scale_w=1.0, scale_alpha=1.0, scale_chord=1.0, show_airfoil=False,save_path='animations/model_opt.gif')




# %%______''' Test de la fonction _phase_align_column'''____________________

# from ROM import _phase_align_column
# import numpy as np
# import matplotlib.pyplot as plt
# avant = np.array([
#     -2.93409564e-04 - 3.14411317e-02j,
#      1.13616867e-07 - 4.13653348e-05j,
#      3.37975911e-09 - 9.35489643e-07j,
#     -3.67894319e-04 - 8.58101896e-02j,
#      4.78765260e-06 + 2.21947399e-03j,
#     -2.06706323e-07 - 1.20589727e-04j
# ])
# apres,k0,arg = _phase_align_column(avant) # sortie de ta fonction

# print(f"Composante de ref k0 = {k0}, déphasage appliqué (rad) = {arg:.6f}")

# # --- Magnitude --- le module n'est évidemment pas changé
# plt.figure()
# plt.plot(np.abs(avant), label='Avant (|Vq[:, i]|)')
# plt.plot(np.abs(apres), linestyle='--', label='Après (|vi|)')
# plt.xlabel('Indice k')
# plt.ylabel('Amplitude')
# plt.title('Amplitude avant / après alignement de phase')
# plt.legend()
# plt.grid(True)

# # --- Phase (déroulée) ---
# plt.figure()
# phase_avant = np.unwrap(np.angle(avant))
# phase_apres = np.unwrap(np.angle(apres))
# plt.plot(phase_avant, label='Avant (phase déroulée)')
# plt.plot(phase_apres, linestyle='--', label='Après (phase déroulée)')
# plt.xlabel('Indice k')
# plt.ylabel('Phase [rad]')
# plt.title('Phase avant / après alignement')
# plt.legend()
# plt.grid(True)

# # --- Plan complexe (nuage de points) ---
# plt.figure()
# plt.plot(avant.real, avant.imag, 'o', label='Avant')
# plt.plot(apres.real, apres.imag, 'x', label='Après')
# plt.xlabel('Réel')
# plt.ylabel('Imaginaire')
# plt.axis('equal')
# plt.title('Plan complexe : avant vs après')
# plt.legend()
# plt.grid(True)

# plt.show()

