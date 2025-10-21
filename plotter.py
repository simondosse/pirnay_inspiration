#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from data_manager import _load_npz


def plot_modal_data_single(npz_path='data/model_params_Theodorsen.npz'):
    """
    Trace un seul jeu de données (fréquences et amortissements) avec un style fixé.

    Règles :
    - Theodorsen : ligne pleine
    - QuasiSteady : ligne pointillée
    - 1ère colonne (mode 1) : bleu ('T1')
    - 2ème colonne (mode 2) : rouge ('B2')
    - Légende unique sur le subplot d'amortissement : '<mode> - <modèle>'
    """
    # Chargement
    D = _load_npz(npz_path)
    U, f, z, p = D['U'], D['f'], D['damping'], D['params']

    # Nom du modèle & style
    name = str(p['model_aero']).lower()
    model_name = 'Theodorsen' if name.startswith('theod') else 'QuasiSteady'
    linestyle = '-' if model_name == 'Theodorsen' else '--'

    # Couleurs et labels par mode
    colors = ['blue', 'red']      # 0 -> T1, 1 -> B2
    mode_labels = ['2nd mode', '3rd mode']

    # Figure
    fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)

    # Fréquences
    for j in (0, 1):
        if not np.all(np.isnan(f[:, j])):
            ax[0].plot(U, f[:, j], color=colors[j], linestyle=linestyle, lw=1.2)
    ax[0].set_ylabel('f [Hz]')
    ax[0].grid(True, linewidth=0.3, alpha=0.5)

    # Amortissement + légende
    for j in (0, 1):
        ax[1].plot(U, z[:, j], color=colors[j], linestyle=linestyle, lw=1.2, label=f"{mode_labels[j]}")


    ax[1].set_xlabel('U [m/s]')
    ax[1].set_ylabel('zeta [-]')
    ax[1].grid(True, linewidth=0.3, alpha=0.5)
    ax[1].legend(frameon=False, ncols=2)
    plt.show()

def plot_modal_data_two(npz_path_a='data/model_params_Theodorsen.npz',
                        npz_path_b='data/model_params_QuasiSteady.npz'):
    """
    
    Plot two simulations together with fixed styling.

    Rules:
    - Theodorsen: solid line
    - QuasiSteady: dashed line
    - First column (mode 1): blue ('T1')
    - Second column (mode 2): red ('B2')
    - Single legend on the damping subplot combining mode and model info
    """
    # Load both datasets
    A = _load_npz(npz_path_a)
    B = _load_npz(npz_path_b)

    Ua, fa, za, pa = A['U'], A['f'], A['damping'], A['params']
    Ub, fb, zb, pb = B['U'], B['f'], B['damping'], B['params']

    U = Ua  # assuming Ua and Ub are the same

    # Determine which dataset is Theodorsen/QuasiSteady and assign styles
    name_a = str(pa['model_aero']).lower() # we get the model_aero parameter from the saved params dict, convert it to string and lowercase it
    name_b = str(pb['model_aero']).lower() # lower() to put in minuscule
    style_a = '-' if name_a.startswith('theod') else '--'
    style_b = '-' if name_b.startswith('theod') else '--'

    # Ensure arrays have up to 2 columns (T1, B2). If only 1, pad with NaN.
    # def take_two(arr: np.ndarray) -> np.ndarray:
    #     if arr.ndim == 1:
    #         return np.column_stack([arr, np.full_like(arr, np.nan)])
    #     if arr.shape[1] == 1:
    #         return np.column_stack([arr[:, 0], np.full(arr.shape[0], np.nan)])
    #     return arr[:, :2]

    # fa2, fb2 = take_two(fa), take_two(fb)
    # za2, zb2 = take_two(za), take_two(zb)

    #------ Fixed colors and labels per mode---
    colors = ['blue', 'red']  # 0 -> T1, 1 -> B2
    # mode_labels = ['B2', 'T1']
    mode_labels = ['2nd mode', '3rd mode']
    #------------------------------------------

    # Create figure with shared x-axis
    fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)

    # Top subplot: frequencies
    for j in (0, 1):
        ax[0].plot(U, fa[:, j], color=colors[j], linestyle=style_a, lw=1.2) #lw for line width
        ax[0].plot(U, fb[:, j], color=colors[j], linestyle=style_b, lw=1.2)
    ax[0].set_ylabel('f [Hz]')
    ax[0].grid(True, linewidth=0.3, alpha=0.5) #alpha for transparency

    # Bottom subplot: damping + legend here
    for j in (0, 1):

        # Legend entries combine mode and model info
        model_a = 'Theodorsen' if style_a == '-' else 'QuasiSteady'
        model_b = 'Theodorsen' if style_b == '-' else 'QuasiSteady'
        # labels += [f"{mode_labels[j]} - {model_a}", f"{mode_labels[j]} - {model_b}"]

        ax[1].plot(U, za[:, j], color=colors[j], linestyle=style_a, lw=1.2, label = mode_labels[j])
        ax[1].plot(U, zb[:, j], color=colors[j], linestyle=style_b, lw=1.2, label = mode_labels[j])

    ax[1].set_xlabel('U [m/s]')
    ax[1].set_ylabel('zeta [-]')
    ax[1].grid(True, linewidth=0.3, alpha=0.5)
    ax[1].legend(frameon=False, ncols=2)

    plt.show()

def plot_params_table(npz_path: str):
    """
    Load a .npz file (via _load_npz) and display its parameters as a table.

    Each parameter name and value is shown in a simple matplotlib table.
    """
    # Load file using your existing loader
    data = _load_npz(npz_path)
    params = data['params']

    if not params:
        print("No parameters found in this file.")
        return

    # Convert to 2D list of [key, value] for the table
    table_data = [[k, str(v)] for k, v in params.items()]

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(6, len(params)*0.4 + 1))
    ax.axis('off')  # no axes, just the table

    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=["Parameter", "Value"],
        loc='center',
        cellLoc='left'
    )

    # Style adjustments
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)  # slightly bigger cells
    ax.set_title(f"Parameters from {npz_path}", fontsize=11, pad=10)

    plt.tight_layout()
    plt.show()

def plot_mode_shapes_grid(y, freqs_hz, W=None, ALPHA=None,extras=None,normalize='per_mode',colors=None,styles=None,sharey=True,figsize=None,suptitle=None,show=True):
    '''
    Trace les formes modales par mode, en colonnes :
    | Mode 1 (f=...) | Mode 2 (f=...) | Mode 3 (f=...) | ...

    Chaque subplot (colonne) superpose w_i(y), alpha_i(y) et, si fourni, des champs supplémentaires (extras).

    Paramètres
    ----------
    y : (Ny,) array
        Abscisses spanwise.
    freqs_hz : (n_modes,) array
        Fréquences par mode (Hz).
    W : (n_modes, Ny) or None
        Formes en flexion w_i(y).
    ALPHA : (n_modes, Ny) or None
        Formes en torsion alpha_i(y).
    extras : dict[str, np.ndarray] or None
        Champs additionnels par mode, ex. {'v': V} avec V shape (n_modes, Ny).
    normalize : {'per_mode','per_field', None}
        - 'per_mode'  : normalise toutes les courbes d’un même mode par le max absolu parmi les champs présents
        - 'per_field' : normalise chaque champ indépendamment (par son propre max absolu)
        - None        : pas de normalisation
    colors : dict[str, str] or None
        Couleurs par champ, ex. {'w': 'C0','alpha': 'C1','v':'C2'}.
    styles : dict[str, str] or None
        Styles de ligne par champ, ex. {'w':'-','alpha':'--','v':':' }.
    sharey : bool
        Partage de l’axe Y entre subplots.
    figsize : tuple or None
        Taille figure (L, H). Défaut calculé sur le nb de modes.
    suptitle : str or None
        Titre global de la figure.
    show : bool
        Appelle plt.show() si True.

    Retour
    ------
    fig, axes : matplotlib Figure et Axes
    '''


    # Construire la collection de champs à tracer
    fields = []
    if W is not None:
        fields.append(('w', np.asarray(W)))
    if ALPHA is not None:
        fields.append(('alpha', np.asarray(ALPHA)))
    if extras:
        for name, mat in extras.items():
            fields.append((str(name), np.asarray(mat)))

    if len(fields) == 0:
        raise ValueError("Aucun champ fourni (W, ALPHA ou extras).")

    # Vérifications de dimensions et harmonisation
    y = np.asarray(y).ravel()
    Ny = y.size

    # Nombre de modes à tracer = min(nb colonnes disponibles, len(freqs_hz))
    n_modes_available = [f[1].shape[0] for f in fields]
    n_modes = int(np.min([np.min(n_modes_available), np.asarray(freqs_hz).size]))

    # Vérifie la dimension Ny
    for name, mat in fields:
        if mat.shape[1] != Ny:
            raise ValueError(f"Le champ '{name}' a Ny={mat.shape[1]} différent de len(y)={Ny}.")

    # Couleurs / styles par défaut
    if colors is None:
        colors = {}
    if styles is None:
        styles = {}

    default_palette = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    default_styles = ['-', '--', ':', '-.']

    # Assigner des couleurs/styles par champ s’ils manquent
    for idx, (name, _) in enumerate(fields):
        colors.setdefault(name, default_palette[idx % len(default_palette)])
        styles.setdefault(name, default_styles[idx % len(default_styles)])

    # Figure
    if figsize is None:
        figsize = (max(5.0, 3.0 * n_modes), 3.2)  # largeur ~3 par mode
    fig, axes = plt.subplots(1, n_modes, sharey=sharey, figsize=figsize, constrained_layout=True)
    if n_modes == 1:
        axes = np.array([axes])

    # Boucle sur les modes (colonnes)
    for i in range(n_modes):
        ax = axes[i]

        # Prépare normalisation
        if normalize == 'per_mode':
            # max absolu sur tous les champs présents pour ce mode i
            acc = []
            for _, mat in fields:
                acc.append(np.max(np.abs(mat[i, :])))
            scale = np.max(acc) if len(acc) > 0 else 1.0
            if scale == 0:
                scale = 1.0

        # Traces des champs
        for name, mat in fields:
            curve = np.array(mat[i, :], dtype=float)

            if normalize == 'per_field':
                s = np.max(np.abs(curve))
                if s > 0:
                    curve = curve / s
            elif normalize == 'per_mode':
                curve = curve / scale

            ax.plot(y, curve, color=colors[name], linestyle=styles[name], lw=1.3, label=name)

        # Titres / axes
        fi = float(freqs_hz[i])
        ax.set_title(f"Mode {i+1} (f={fi:.2f} Hz)")
        if i == 0:
            ax.set_ylabel("Amplitude [a.u.]")
        ax.set_xlabel("y [m]")
        ax.grid(True, linewidth=0.3, alpha=0.5)

        # Légende sur le dernier subplot uniquement (évite la répétition)
        if i == n_modes - 1:
            ax.legend(frameon=False)

    if suptitle:
        fig.suptitle(suptitle, y=1.02)

    if show:
        plt.show()

    return fig, axes


