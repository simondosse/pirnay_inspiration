
import os
import numpy as np

import ROM

def _load_npz(npz_path: str):
    """Load a saved .npz and return arrays and params.

    Returns a dict with keys: 'U', 'f', 'damping', 'params'.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}. Run main.py first to generate it.")
    data = np.load(npz_path, allow_pickle=True)
    params = {k[2:]: data[k] for k in data.files if k.startswith('p_')}
    return {'U': data['U'], 'f': data['f'], 'damping': data['damping'], 'params': params}

def save_modal_data(f, damping, model_params, out_dir='data', filename='model_params.npz'):
    """
    Run the Reduced Order Model once and save outputs to a .npz file.

    Saves arrays: U, f, damping
    Returns the created file path.
    """

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    # Run the ROM to produce outputs to save
    

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