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


def run_model(model_params):  # run the model to get f and the damping
    """
    Run the Reduced Order Model once and return results.
    """
    f, damping, _ = ROM.ModalParamDyn(model_params)
    return f, damping


def save_modal_data(model_params, out_dir='data', filename='model_params.npz'):
    """
    Run the Reduced Order Model once and save outputs to a .npz file.

    Saves arrays: U, f, damping
    Returns the created file path.
    """

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    # Run the ROM to produce outputs to save
    f, damping, _ = ROM.ModalParamDyn(model_params)

    # Save compactly to .npz (NumPy archive)
    params = dict(model_params.__dict__)
    # Remove arrays that will be saved explicitly
    for k in ('U',): # should we add y ?
        params.pop(k, None) # remove the arrays already saved separately (in the same .npz but not in the dict)

    # Prefix keys to avoid collisions
    params_prefixed = {f"p_{k}": np.asarray(v) for k, v in params.items()}
    # we recreate a new dict with the same values but the keys have a prefix 'p_'
    # mass : p_mass, just to say it's parameter and not an output like U, f or damping

    # save all data to .npz
    np.savez(
        out_path,
        U=model_params.U,
        f=f,
        damping=damping,
        **params_prefixed, #we also save all the parameters with the prefix
    )
    print(f"Saved modal data to {out_path}")
    return out_path





def main():
    # Build base structural parameters (common to both runs)

    ''' Structural parameters '''
    s = 2 #half span
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


    model_theod = ModelParameters(
        s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t, 'Theodorsen'
    )
    save_modal_data(model_theod, filename='model_params_TheodorsenS2.npz')

    model_qs = ModelParameters(
        s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t, 'QuasiSteady'
    )
    save_modal_data(model_qs, filename='model_params_QuasiSteadyS2.npz')

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


if __name__ == "__main__":
    main()
