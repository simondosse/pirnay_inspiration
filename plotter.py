#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from data_manager import _load_npz
from typing import Tuple, Optional 

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
    # Robust to params saved with private name '_model_aero'
    name = str(p.get('model_aero', p.get('_model_aero', 'Theodorsen'))).lower()
    model_name = 'Theodorsen' if name.startswith('theod') else 'QuasiSteady'
    linestyle = '-' if model_name == 'Theodorsen' else '--'

    # Nombre de modes (colonnes)
    n_modes = int(f.shape[1]) if (hasattr(f, 'ndim') and f.ndim == 2) else 1

    # Couleurs et labels par mode (génériques)
    # Utilise le cycle de couleurs matplotlib si disponible
    try:
        base_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    except Exception:
        base_colors = []
    if len(base_colors) < n_modes:
        colors = [f"C{i}" for i in range(n_modes)]
    else:
        colors = base_colors[:n_modes]
    mode_labels = [f"Mode {j+1}" for j in range(n_modes)]

    # Figure
    fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    fig.suptitle(npz_path)

    # Fréquences
    if n_modes == 1 and (not hasattr(f, 'ndim') or f.ndim == 1):
        if not np.all(np.isnan(f)):
            ax[0].plot(U, f, color=colors[0], linestyle=linestyle, lw=1.2)
    else:
        for j in range(n_modes):
            if not np.all(np.isnan(f[:, j])):
                ax[0].plot(U, f[:, j], color=colors[j], linestyle=linestyle, lw=1.2)
    ax[0].set_ylabel('f [Hz]')
    ax[0].grid(True, linewidth=0.3, alpha=0.5)

    # Amortissement + légende
    if n_modes == 1 and (not hasattr(z, 'ndim') or z.ndim == 1):
        ax[1].plot(U, z, color=colors[0], linestyle=linestyle, lw=1.2, label=mode_labels[0])
    else:
        for j in range(n_modes):
            ax[1].plot(U, z[:, j], color=colors[j], linestyle=linestyle, lw=1.2, label=mode_labels[j])

    ax[1].set_xlabel('U [m/s]')
    ax[1].set_ylabel('zeta [-]')
    ax[1].grid(True, linewidth=0.3, alpha=0.5)
    ax[1].legend(frameon=False, ncols=min(4, n_modes))
    plt.show()

def plot_modal_data_two(npz_path_a,
                        npz_path_b):
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


    style_a = '-'
    style_b = '--'

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
    mode_labels = ['B2', 'T1']
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

        # labels += [f"{mode_labels[j]} - {model_a}", f"{mode_labels[j]} - {model_b}"]

        ax[1].plot(U, za[:, j], color=colors[j], linestyle=style_a, lw=1.2, label = mode_labels[j] +' '+ npz_path_a.rsplit('_',1)[-1])
        ax[1].plot(U, zb[:, j], color=colors[j], linestyle=style_b, lw=1.2, label = mode_labels[j] +' '+ npz_path_b.rsplit('_',1)[-1])

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

def plot_vi(vi: np.ndarray,
            Nw: int,
            Nalpha: int,
            kind: str = 'abs',              # 'abs' | 'real_imag' | 'mag_phase'
            normalize: Optional[str] = None,# None | 'max' | 'l2'
            align_phase: bool = False,
            ref_idx: Optional[int] = None,
            figsize: Tuple[int, int] = (9, 4),
            bending_label: str = 'vw (bending)',
            torsion_label: str = 'va (torsion)',
            ax: Optional[plt.Axes] = None   # NEW: draw into an existing axis
):
    vi = np.asarray(vi).reshape(-1)
    assert vi.size == Nw + Nalpha, "vi must have length Nw+Nalpha"
    v_plot = vi.copy()

    # Optional phase alignment
    if align_phase and v_plot.size > 0:
        if ref_idx is None:
            ref_idx = int(np.argmax(np.abs(v_plot)))
        ang = np.angle(v_plot[ref_idx])
        v_plot = v_plot * np.exp(-1j * ang)

    # normalization across the full vector (vw+va)
    if normalize == 'max':
        a = np.max(np.abs(v_plot)) or 1.0
        v_plot = v_plot / a
    elif normalize == 'l2':
        a = np.sqrt(np.vdot(v_plot, v_plot).real) or 1.0
        v_plot = v_plot / a

    vw = v_plot[:Nw]
    va = v_plot[Nw:]

    # mag_phase needs two axes -> only allowed when ax is None
    if kind == 'mag_phase' and ax is not None:
        raise ValueError("kind='mag_phase' requires ax=None (creates a 2-row figure)")

    if kind == 'mag_phase':
        fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
        ax_mag, ax_phi = axes
        ax_mag.bar(np.arange(Nw), np.abs(vw), color='#4C78A8', label=bending_label)
        ax_mag.bar(Nw + np.arange(Nalpha), np.abs(va), color='#F58518', label=torsion_label)
        ax_mag.set_xticks(np.arange(Nw + Nalpha))
        ax_mag.set_xticklabels([f"w{k+1}" for k in range(Nw)] + [f"α{k+1}" for k in range(Nalpha)])
        ax_mag.set_ylabel("|coeff|"); ax_mag.legend(); ax_mag.set_title("Modal coefficients magnitude")

        phase = np.angle(np.concatenate([vw, va]))
        ax_phi.plot(np.arange(Nw + Nalpha), phase, 'o-', color='#6F4E7C')
        ax_phi.axhline(0.0, color='k', lw=0.8, alpha=0.5)
        ax_phi.set_xticks(np.arange(Nw + Nalpha))
        ax_phi.set_xticklabels([f"w{k+1}" for k in range(Nw)] + [f"α{k+1}" for k in range(Nalpha)])
        ax_phi.set_ylabel("phase [rad]"); ax_phi.set_title("Modal coefficients phase")
        return fig, axes

    # Single-axis variants
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        created_fig = True

    idx_w = np.arange(Nw)
    idx_a = Nw + np.arange(Nalpha)

    if kind == 'abs':
        ax.bar(idx_w, np.abs(vw), color='#4C78A8', label=bending_label)
        ax.bar(idx_a, np.abs(va), color='#F58518', label=torsion_label)
        ax.set_ylabel("|coeff|")
        ax.set_title("Modal coefficients (magnitude)")
    elif kind == 'real_imag':
        width = 0.38
        ax.bar(idx_w - width/2, vw.real, width, color='#4C78A8', label=f"Re {bending_label}")
        ax.bar(idx_w + width/2, vw.imag, width, color='#72B7B2', label=f"Im {bending_label}")
        ax.bar(idx_a - width/2, va.real, width, color='#F58518', label=f"Re {torsion_label}")
        ax.bar(idx_a + width/2, va.imag, width, color='#E45756', label=f"Im {torsion_label}")
        ax.set_ylabel("value")
        ax.set_title("Modal coefficients (real/imag)")
        ax.axhline(0.0, color='k', lw=0.8, alpha=0.5)
    else:
        raise ValueError("kind must be 'abs', 'real_imag', or 'mag_phase'")

    ax.set_xticks(np.arange(Nw + Nalpha))
    ax.set_xticklabels([f"w{k+1}" for k in range(Nw)] + [f"α{k+1}" for k in range(Nalpha)])
    ax.grid(True, linewidth=0.3, alpha=0.5)

    if created_fig:
        ax.legend()
        return fig, ax
    else:
        return ax


def plot_vi_grid(Vq: np.ndarray,Nw: int,Nalpha: int,
                 freqs_hz: Optional[np.ndarray] = None,
                 kind: str = 'abs',              # 'abs' | 'real_imag'
                 normalize: Optional[str] = 'l2',# None | 'max' | 'l2' (per mode)
                 align_phase: bool = False,
                 sharey: bool = True,
                 figsize: Optional[Tuple[float, float]] = None,
                 suptitle: Optional[str] = None,
                 show: bool = True,
                 bending_label: str = 'vw (bending)',
                 torsion_label: str = 'va (torsion)',
):
    Vq = np.asarray(Vq)
    assert Vq.shape[0] == Nw + Nalpha, "Vq must have Nw+Nalpha rows"
    n_modes = Vq.shape[1]

    if figsize is None:
        figsize = (max(5.0, 3.0 * n_modes), 3.2)

    fig, axes = plt.subplots(1, n_modes, sharey=sharey, figsize=figsize, constrained_layout=True)
    if n_modes == 1:
        axes = np.array([axes])

    for i in range(n_modes):
        ax = axes[i]
        plot_vi(
            vi=Vq[:, i],
            Nw=Nw, Nalpha=Nalpha,
            kind=kind,
            normalize=normalize,
            align_phase=align_phase,
            bending_label=bending_label,
            torsion_label=torsion_label,
            ax=ax,  # reuse single-axis variant
        )
        title = f"Mode {i+1}"
        if freqs_hz is not None:
            title += f" (f={float(freqs_hz[i]):.2f} Hz)"
        ax.set_title(title)

        if i == n_modes - 1:
            ax.legend(frameon=False)

    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes

def plot_mode_shapes_grid(y, freqs_hz, W=None, ALPHA=None,extras=None,normalize=False,colors=None,styles=None,sharey=True,figsize=None,suptitle=None,show=True):
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
        - 'per_mode'  : normalise toutes les courbes d'un même mode par le max absolu parmi les champs présents
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


    En vrai l'arg "normalize" ne sert à rien comme on traite l'amplitude des ces vecteurs en amont
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
        figsize = (max(5.0, 3.0 * n_modes), 3.4)  # largeur ~3 par mode
    fig, axes = plt.subplots(1, n_modes, sharey=sharey, figsize=figsize, constrained_layout=True)
    if n_modes == 1:
        axes = np.array([axes])

    # Boucle sur les modes (colonnes)
    for i in range(n_modes):
        ax = axes[i]
        # Traces des champs
        for name, mat in fields:
            curve = np.array(mat[i, :], dtype=float)
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
    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes

def plot_mode_shapes_over_U_grid(y, U, WU=None, ALPHAU=None, f_modes_U=None,
                                 mode_indices=None, n_samples=10,
                                 colors=None, styles=None,
                                 sharey=True, figsize=None,
                                 suptitle=None, show=True,):
    """
    Plot spatial mode shapes (w and alpha) for multiple wind speeds in a grid.

    Inputs
    ------
    y : array-like, shape (Ny,)
        Spanwise coordinate array.
    U : array-like, shape (nU,)
        Wind speed samples corresponding to WU/ALPHAU.
    WU : array-like or None, shape (nU, n_modes, Ny)
        Bending shapes w_i(y) reconstructed at each U. If None, only alpha is plotted.
    ALPHAU : array-like or None, shape (nU, n_modes, Ny)
        Torsion shapes alpha_i(y) reconstructed at each U. If None, only w is plotted.
    f_modes_U : array-like or None, shape (nU, n_modes)
        Modal frequencies (Hz) per mode and U. If provided, each subplot
        is annotated with its corresponding frequency.
    mode_indices : list[int] or None
        Modes to plot. Defaults to all available modes.
        - Accepts 0-based indices (e.g., [0, 1, 2]).
        - Also accepts 1-based indices (e.g., [1, 2, 3]); detection is automatic:
        if all(idx >= 1) and max(idx) <= n_modes, they are treated as 1-based.
    n_samples : int
        Number of U samples to plot, evenly spaced from U[0] to U[-1]. Clipped to len(U).
    colors : dict[str, str] or None
        Colors per field name. Defaults: {'w':'C0', 'alpha':'C1'}.
    styles : dict[str, str] or None
        Line styles per field name. Defaults: {'w':'-', 'alpha':'--'}.
    sharey : bool
        Share Y axis across subplots. Keep False if you already normalized upstream.
    figsize : (float, float) or None
        Figure size. Defaults to (3.0 * n_modes, 2.6 * n_rows), clamped to reasonable minimums.
    suptitle : str or None
        Global figure title.
    show : bool
        If True, calls plt.show().

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array of shape (n_rows, n_cols)

    Notes
    -----
    - WU/ALPHAU are expected as (nU, n_modes, Ny) from your ModalParamDyn.
    - Each subplot overlays w and alpha for a single (mode, U) pair.
    Columns = modes; rows = selected U’s.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if WU is None and ALPHAU is None:
        raise ValueError("Provide at least one of WU or ALPHAU.")

    if WU is not None:
        WU = np.asarray(WU)
    if ALPHAU is not None:
        ALPHAU = np.asarray(ALPHAU)

    U = np.asarray(U).ravel()
    y = np.asarray(y).ravel()
    nU = U.size
    Ny = y.size

    # Infer n_modes and validate shapes
    n_modes_candidates = []
    if WU is not None:
        if WU.ndim != 3:
            raise ValueError("WU must have shape (nU, n_modes, Ny).")
        if WU.shape[0] != nU or WU.shape[2] != Ny:
            raise ValueError(f"WU shape mismatch: got {WU.shape}, expected (nU={nU}, n_modes, Ny={Ny}).")
        n_modes_candidates.append(WU.shape[1])
    if ALPHAU is not None:
        if ALPHAU.ndim != 3:
            raise ValueError("ALPHAU must have shape (nU, n_modes, Ny).")
        if ALPHAU.shape[0] != nU or ALPHAU.shape[2] != Ny:
            raise ValueError(f"ALPHAU shape mismatch: got {ALPHAU.shape}, expected (nU={nU}, n_modes, Ny={Ny}).")
        n_modes_candidates.append(ALPHAU.shape[1])
    if f_modes_U is not None:
        f_modes_U = np.asarray(f_modes_U)
        if f_modes_U.ndim != 2 or f_modes_U.shape[0] != nU:
            raise ValueError(
                f"f_modes_U must have shape (nU, n_modes). Got {getattr(f_modes_U, 'shape', None)} with nU={nU}."
            )
        n_modes_candidates.append(f_modes_U.shape[1])

    if not n_modes_candidates:
        raise ValueError("Cannot infer number of modes. Provide WU and/or ALPHAU with valid shapes.")
    n_modes_total = int(min(n_modes_candidates))  # safe choice if shapes differ slightly

    # Select modes
    if mode_indices is None:
        mode_indices_0 = list(range(n_modes_total))
    else:
        idx = np.asarray(mode_indices, dtype=int).ravel().tolist()
        if len(idx) > 0 and min(idx) >= 1 and max(idx) <= n_modes_total:
            mode_indices_0 = [k - 1 for k in idx]  # convert 1-based to 0-based
        else:
            mode_indices_0 = idx
        for k in mode_indices_0:
            if k < 0 or k >= n_modes_total:
                raise ValueError(f"Mode index {k} out of range [0, {n_modes_total-1}].")

    n_cols = len(mode_indices_0)
    if n_cols == 0:
        raise ValueError("No modes selected to plot.")

    # Pick evenly spaced U indices
    n_rows = int(min(max(1, n_samples), nU))
    idx_rows = np.linspace(0, nU - 1, n_rows, dtype=int)
    idx_rows = np.unique(idx_rows)
    n_rows = idx_rows.size

    # Colors / styles defaults
    if colors is None:
        colors = {}
    if styles is None:
        styles = {}
    colors.setdefault('w', 'C0')
    colors.setdefault('alpha', 'C1')
    styles.setdefault('w', '-')
    styles.setdefault('alpha', '--')

    # Figure sizing
    if figsize is None:
        figsize = (max(5.0, 3.0 * n_cols), max(3.0, 1.3 * n_rows))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        sharey=sharey,
        figsize=figsize,
        constrained_layout=True
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # Plot
    for r, iu in enumerate(idx_rows):
        Uval = float(U[iu])
        for c, kmode in enumerate(mode_indices_0):
            ax = axes[r, c]

            # w field
            if WU is not None:
                ax.plot(
                    y, np.asarray(WU[iu, kmode, :], dtype=float),
                    color=colors.get('w', 'C0'),
                    linestyle=styles.get('w', '-'),
                    lw=1.3,
                    label='w'
                )
            # alpha field
            if ALPHAU is not None:
                ax.plot(
                    y, np.asarray(ALPHAU[iu, kmode, :], dtype=float),
                    color=colors.get('alpha', 'C1'),
                    linestyle=styles.get('alpha', '--'),
                    lw=1.3,
                    label='alpha'
                )

            # Titles / labels
            if r == 0:
                ax.set_title(f"Mode {kmode+1}")
            if c == 0:
                ax.set_ylabel(f"U = {Uval:.2f} m/s")
            if r == n_rows - 1:
                ax.set_xlabel("y [m]")

            ax.grid(True, linewidth=0.3, alpha=0.5)

            # Annotate frequency if provided
            if f_modes_U is not None:
                try:
                    fval = float(f_modes_U[iu, kmode])
                    if np.isfinite(fval):
                        ax.text(
                            0.98, 0.06, f"f = {fval:.2f} Hz",
                            transform=ax.transAxes,
                            ha='right', va='bottom', fontsize=9, color='0.35'
                        )
                except Exception:
                    pass

            # Legend only on the last subplot
            if (r == n_rows - 1) and (c == n_cols - 1):
                ax.legend(frameon=False)

    if suptitle:
        fig.suptitle(suptitle, y=1.02)

    if show:
        import matplotlib.pyplot as plt
        plt.show()

    return fig, axes

def animate_beam(par,t,X,U=None,n_stations=15,interval=30,
                 scale_w=1.0,scale_alpha=1.0,scale_chord=1.0,
                 show_airfoil=False,airfoil_points=60,
                 x_ref='EA',            # 'EA' ou 'CG' comme point de rotation visuelle
                 ylim=None,
                 xlim=None,
                 save_path=None,        # '.mp4' ou '.gif' si tu veux enregistrer
                 repeat=True,
):
    """
    Anime la poutre à partir de X(t): ligne d'axe w(y,t) + cordes locales orientées par alpha(y,t).

    - par.airfoil fournit la géométrie NACA (c, x_ea, x_cg, h(x)).
    - On projette en 2D (plan y–z). La torsion crée un décalage vertical ~ (x - x_ea)*sin(alpha).

    Paramètres
    ----------
    par : ModelParameters (cf. input.py)
    t : (nt,) array
    X : (nt, 2*(Nv+Nw+Nalpha)) array (sortie integrate_state_rk)
    U : float ou None (annot)
    n_stations : nb de stations spanwise où dessiner les cordes locales
    interval : ms entre frames (matplotlib.animation)
    scale_w : facteur d’échelle sur w (visuel)
    scale_alpha : facteur d’échelle sur alpha (visuel)
    scale_chord : facteur d’échelle visuel de la corde (longueur affichée)
    show_airfoil : si True, projette un petit contour 2D (haut/bas) à chaque station
    airfoil_points : nb de points le long de la corde pour le contour 2D
    x_ref : 'EA' ou 'CG' (point de pivot pour la visualisation)
    ylim, xlim : limites axes (z et y)
    save_path : chemin pour enregistrer (None pour ne pas enregistrer)
    repeat : boucle l’animation

    Retour
    ------
    fig, ani
    """
    from ROM import _modal_to_physical_fields, X_to_q  # import local pour éviter cycles

    t = np.asarray(t).ravel()
    nt = t.size
    if X.shape[0] != nt:
        raise ValueError("X and t must have the same dimensions.")

    # we get the generalized coordinates from the state vector X(t)
    qw_t, qa_t = X_to_q(par = par, X=X, t=t)

    # Reconstruction champs (nt, Ny)
    w_map, a_map = _modal_to_physical_fields(par, qw_t, qa_t)
    y = np.asarray(par.y, dtype=float).ravel()
    Ny = y.size
    if w_map.shape[0] != nt or w_map.shape[1] != Ny:
        raise ValueError("Dimensions de w_map inattendues.")
    if a_map.shape[0] != nt or a_map.shape[1] != Ny:
        raise ValueError("Dimensions de a_map inattendues.")

    # Géométrie NACA
    af = par.airfoil
    c = float(af.c)
    x_ea = float(af.x_ea)
    x_cg = float(af.x_cg)
    pivot_x = x_ea if (str(x_ref).upper() == 'EA') else x_cg

    # Stations pour dessiner les cordes
    n_st = int(max(2, min(n_stations, Ny)))
    idx_st = np.linspace(0, Ny - 1, n_st, dtype=int)
    y_st = y[idx_st]

    # Prépare figure
    fig, ax = plt.subplots(figsize=(10, 4))
    title = ax.set_title("Structural animation")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    if xlim is None:
        ax.set_xlim(y[0], y[-1])
    else:
        ax.set_xlim(*xlim)

    # Estimation basique des bornes z si non données
    z_est = scale_w * np.max(np.abs(w_map))
    chord_span = scale_chord * (c) * 0.5  # excursion verticale max approx due à torsion
    z_pad = 0.1 * max(1.0, z_est + chord_span)
    if ylim is None:
        ax.set_ylim(-z_est - chord_span - z_pad, z_est + chord_span + z_pad)
    else:
        ax.set_ylim(*ylim)

    # Ligne d'axe (w)
    center_line, = ax.plot([], [], 'k-', lw=1.8, label='centerline w(y,t)')

    # Cordes locales (LE–TE) + marqueurs EA/CG
    chord_lines = []
    ea_points = []
    cg_points = []
    for _ in idx_st:
        ln, = ax.plot([], [], color='tab:blue', lw=1.2, alpha=0.9)
        chord_lines.append(ln)
        ea, = ax.plot([], [], 'ro', ms=3, alpha=0.8)  # EA
        ea_points.append(ea)
        cg, = ax.plot([], [], 'go', ms=3, alpha=0.8)  # CG
        cg_points.append(cg)

    # Option: petit contour airfoil projeté (yz)
    af_lines = []
    if show_airfoil:
        x_samp = np.linspace(0.0, c, int(max(airfoil_points, 20)))
        z_upper = +np.interp(x_samp, af.x, af.h)  # h(x) déjà en mètres
        z_lower = -z_upper
        for _ in idx_st:
            up, = ax.plot([], [], color='0.5', lw=0.8, alpha=0.6)
            lo, = ax.plot([], [], color='0.5', lw=0.8, alpha=0.6)
            af_lines.append((up, lo))
    else:
        x_samp = None
        z_upper = None
        z_lower = None

    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc='upper right', frameon=False)

    def init():
        center_line.set_data([], [])
        for ln in chord_lines:
            ln.set_data([], [])
        for p in ea_points + cg_points:
            p.set_data([], [])
        if show_airfoil:
            for up, lo in af_lines:
                up.set_data([], [])
                lo.set_data([], [])
        return [center_line, *chord_lines, *ea_points, *cg_points] + ([l for pair in af_lines for l in pair] if show_airfoil else [])

    def update(frame):
        w = scale_w * w_map[frame, :]               # (Ny,)
        a = scale_alpha * a_map[frame, :]           # (Ny,)

        # Ligne d’axe
        center_line.set_data(y, w)

        # Cordes locales
        for k, j in enumerate(idx_st):
            yj = y_st[k]
            wj = w[j]
            aj = a[j]

            # Segment LE–TE (projection 2D yz) pivoté autour de pivot_x
            # z(x) = wj + (x - pivot_x) * sin(alpha_j) ; x projeté via scale_chord
            x_le, x_te = 0.0, c
            z_le = wj + scale_chord * (x_le - pivot_x) * np.sin(aj)
            z_te = wj + scale_chord * (x_te - pivot_x) * np.sin(aj)
            chord_lines[k].set_data([yj, yj], [z_le, z_te])

            # EA / CG (z au pivot_x et au x_cg)
            z_ea = wj + scale_chord * (x_ea - pivot_x) * np.sin(aj)
            z_cg = wj + scale_chord * (x_cg - pivot_x) * np.sin(aj)
            ea_points[k].set_data([yj], [z_ea])
            cg_points[k].set_data([yj], [z_cg])

            # Contour airfoil projeté
            if show_airfoil:
                # rotation petite mais on garde sin/cos pour robustesse
                z_up = wj + scale_chord * (x_samp - pivot_x) * np.sin(aj) + z_upper * np.cos(aj)
                z_lo = wj + scale_chord * (x_samp - pivot_x) * np.sin(aj) + z_lower * np.cos(aj)
                # on trace en yz: y = const, z = f(x)
                # pour alléger, on ne montre que l’épaisseur (projection z)
                af_lines[k][0].set_data(np.full_like(x_samp, yj), z_up)
                af_lines[k][1].set_data(np.full_like(x_samp, yj), z_lo)

        # Titre
        if U is not None:
            title.set_text(f"Structural mode animation  |  U = {U} m/s  |  t = {t[frame]:.3f} s")
        else:
            title.set_text(f"Structural mode animation  |  t = {t[frame]:.3f} s")

        return [center_line, *chord_lines, *ea_points, *cg_points] + ([l for pair in af_lines for l in pair] if show_airfoil else [])

    ani = animation.FuncAnimation(
        fig, update, frames=nt, init_func=init, interval=interval, blit=True, repeat=repeat
    )

    if save_path:
        ext = str(save_path).lower().rsplit('.', 1)[-1]
        if ext == 'gif':
            ani.save(save_path, writer='pillow', fps=max(1, int(1000 / interval)))
        else:
            ani.save(save_path, writer='ffmpeg', fps=max(1, int(1000 / interval)))

    return fig, ani