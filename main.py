''' Import the necessary libraries '''
import os
import numpy as np
import matplotlib.pyplot as plt
from input import ModelParameters
import ROM

# plt.close('all')
# plt.ion()

# modifie tous les parametres de matplotlib
# --- Configuration globale Matplotlib ---
plt.rcParams.update({
    'figure.figsize': [7.17, 4.34],       # Taille des figures (format article)
    # 'text.usetex': True,                  # Active le rendu LaTeX
    # 'text.latex.preamble': r'\usepackage{mathptmx}',  # Police Times pour le texte et les maths
    'font.family': 'serif',               # Police avec empattements
    'font.size': 11,                      # Taille globale du texte
    'axes.labelsize': 11,                 # Taille des labels d'axes
})


def run_model(model_params): # run the model to get f and the damping and plot the results
    '''
    Run the Reduced Order Model and plot the results.
    '''

    f, damping ,_ = ROM.ModalParamDyn(model_params)

    # Save results to a .npz file in /data


def save_modal_data(model_params, out_dir='data', filename='modal_params.npz'):
    """
    Run the Reduced Order Model once and save outputs to a .npz file.

    Saves arrays: U, f, damping
    Returns the created file path.
    """
    # Compute modal parameters
    f, damping, _ = ROM.ModalParamDyn(model_params)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    # Save compactly to .npz (NumPy archive)
    np.savez(out_path, U=model_params.U, f=f, damping=damping)
    print(f"Saved modal data to {out_path}")
    return out_path





def main():
    # Example usage of the Reduced Order Model
    
    # these parameters are defined in input.py

    ''' Structural parameters '''
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

    ''' Wingtip parameters '''
    k = 0.001687 # I = kMt

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

    ''' Aerodynamic model '''
    model_aero = 'Theodorsen'  # Options: 'Theodorsen' or 'QuasiSteady'

    model = ModelParameters(s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t, model_aero)


    ''' Parametric studies '''
    x_t_new = np.linspace(0, 0.02, 10)

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

    # Save data for plotting in plotter.py (no plots here)
    save_modal_data(model)

if __name__ == "__main__":
    main()
