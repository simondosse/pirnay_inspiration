import os
import numpy as np
import matplotlib.pyplot as plt


def plot_modal_data(npz_path='data/modal_params.npz'):
    """
    Load modal parameters saved by main.py and plot them.

    Expects arrays in the archive:
      - U:       (N,)
      - f:       (N, M) or (N,)
      - damping: (N, M) or (N,)
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}. Run main.py first to generate it.")

    data = np.load(npz_path)
    U = data['U']
    f = data['f']
    damping = data['damping']

    # Create figure with shared x-axis
    fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    fig.suptitle('Modal parameters vs Wind speed', fontsize=12)

    # Plot each mode if multiple columns are present
    n_f_modes = f.shape[1] if f.ndim == 2 else 1
    n_zeta_modes = damping.shape[1] if damping.ndim == 2 else 1

    for i in range(n_f_modes):
        y = f[:, i] if n_f_modes > 1 else f
        ax[0].plot(U, y, lw=0.9, label=f'f{i+1}')

    for i in range(n_zeta_modes):
        y = damping[:, i] if n_zeta_modes > 1 else damping
        # Use mathtext for zeta to avoid encoding issues
        ax[1].plot(U, y, lw=0.9, label=rf'$\zeta_{{{i+1}}}$')

    # Labels and grids
    ax[0].set_ylabel('f [Hz]')
    ax[0].grid(True, linewidth=0.3, alpha=0.5)

    ax[1].set_xlabel('U [m/s]')
    ax[1].set_ylabel(r'$\zeta$')
    ax[1].grid(True, linewidth=0.3, alpha=0.5)

    # Optional legends if multiple modes
    if n_f_modes > 1:
        ax[0].legend(frameon=False)
    if n_zeta_modes > 1:
        ax[1].legend(frameon=False)

    plt.show()


# if __name__ == '__main__':
#     plot_modal_data()

plot_modal_data()
