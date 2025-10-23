import numpy as np

def map_to_physical(X_uv):
    '''
    Function go to back to physical parameters as we changed variables
    
    Parameters
    ----------
    X_uv : array  (N,np_para) avec colonnes [u_ea, v_cg, EI, GJ]
        list N*nb_para, samples to be evaluated in the UV space
    
    Returns
    ---------
        np.array
        samples that have be evaluated F(X), (N,4) physiques [x_ea/c, x_cg/c, EI, GJ])
    '''
    x_ea_c_low, x_ea_c_high = 0.15 , 0.5    # x_ea/c ∈ [0.15, 0.5], x_ea can't go beyond the half-chord
    x_cg_c_high = 0.8                     # x_cg/c ≤ 0.8
    # idk if we should add a lower boundary for x_cg (sometimes x_cg_low shoudn't be x_ea)

    u     = X_uv[:, 0]
    v     = X_uv[:, 1]
    EIx   = X_uv[:, 2]
    GJ    = X_uv[:, 3]

    x_ea_c = x_ea_c_low + (x_ea_c_high - x_ea_c_low) * u
    x_cg_c = x_ea_c + (x_cg_c_high - x_ea_c) * v

    X = np.column_stack([x_ea_c, x_cg_c, EIx, GJ])
    return X
