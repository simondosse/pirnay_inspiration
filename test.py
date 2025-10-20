
# JUSTE POUR TESTER UN TRUC


#%%
import numpy as np
import matplotlib.pyplot as plt
from input import ModelParameters
import ROM
#%%
# modifie tous les parametres de matplotlib
# --- Configuration globale Matplotlib ---
plt.rcParams.update({
    'figure.figsize': [7.17, 3.34],       # Taille des figures (format article)
    # 'text.usetex': True,                  # Active le rendu LaTeX
    # 'text.latex.preamble': r'\\usepackage{mathptmx}',  # Police Times pour le texte et les maths
    'font.family': 'serif',               # Police avec empattements
    'font.size': 11,                      # Taille globale du texte
    'axes.labelsize': 11,                 # Taille des labels d'axes
})


def run_model(model_params):
    '''
    Run the Reduced Order Model and plot the results.
    '''

    f, damping ,_ = ROM.ModalParamDyn(model_params)

    fig , ax = plt.subplots(2,1,sharex=True, gridspec_kw={'hspace': 0.2})

    for i in range(0,2):
        ax[0].plot(model_params.U , f[:,i], linewidth = 0.6, c = 'b')
        ax[1].plot(model_params.U , damping[:,i], linewidth = 0.6, c = 'b')
  
    ax[0].set_ylabel(r"$f$ [Hz]")
    ax[0].set_xlim(0, model_params.Umax)
    ax[0].grid(True, linewidth=0.3, alpha=0.5)
    ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    
    ax[1].set_xlabel(r"$U$ [m/s]")
    ax[1].set_ylabel(r"$\zeta$ ")
    ax[1].set_xlim(0,model_params.Umax)
    ax[1].set_ylim(bottom = 0)
    ax[1].grid(True, linewidth=0.3, alpha=0.5)
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    
    plt.tight_layout()
    plt.show()


#%%
# Example usage of the Reduced Order Model

# these parameters are defined in input.py
s = 1.5 #half span
c = 0.2 #chord
x_ea = c/3 # elastic axis location from leading edge
x_cg = 0.379*c # center of gravity location from leading edge
m = 2.4 # mass per unit span
I_alpha = 5.6e-3 # mass moment of inertia per unit span
EIx = 366 # bending stiffness
GJ = 78 # torsional stiffness
eta_w = 0.005 # structural damping ratio in bending
eta_alpha = 0.005 # structural damping ratio in torsion
Mt = None   # mass of the tip body
I_alpha_t = None # mass moment of inertia of the tip body
x_t = None  # location of the tip body from leading edge
model_aero = 'Theodorsen'  # Options: 'Theodorsen' or 'QuasiSteady'

model = ModelParameters(s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t, model_aero)


''' Parametric studies '''
# m_new = np.linspace(m * 0.5, m * 1.5, 10)

# for m_val in m_new:
#     model.update(m=m_val)
#     print(f"Updated model with mass: {model.m}")


''' Run the model '''




f, damping ,_ = ROM.ModalParamDyn(model)

# %%
