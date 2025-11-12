import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import kv
import itertools
import matplotlib.pyplot as plt
import sys
''' General functions '''

def _eigs_sorted_positive_imag(A, Nq=None):
    eigvals, eigvecs = np.linalg.eig(A) # est-ce que ça renvoie déjà 

    # normalisation des vecteurs propres ?
    # print(eigvecs)
    # eigvecs = eigvecs / np.linalg.norm(eigvecs,axis=0)
    # print(eigvals)
    # On prend un élément sur deux : un représentant par paire conjuguée,
    # de base np.linalg.eig() nous renvoie les paires à la suite des autres
    eigvals = eigvals[::2]
    eigvecs = eigvecs[:, ::2]

    #
    order = np.argsort(np.abs(np.imag(eigvals)))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Tronque à Nq ??
    if Nq is not None:
        eigvals = eigvals[:Nq]
        eigvecs = eigvecs[:, :Nq]

    return eigvals, eigvecs

def _phase_align_column(vec):
    # Aligne la phase de tout le vecteur en se basant sur la plus grande composante
    # souvent inutile si dernière on travaille avec une val abs ||
    k0 = int(np.argmax(np.abs(vec))) # k0 indice de la composante à la plus grande amplitude

    if vec[k0] != 0:
        return vec * np.exp(-1j * np.angle(vec[k0])), k0, -np.angle(vec[k0])
    else:
        return vec, k0, 0.0
    '''
    on fait tourner le vecteur pour que la composante max soit réelle et positive (puisqu'on a annulé sa phase en le faisant tourner)
    comme ça la multiplication avec Phi_w ou Phi_alpha se fera bien
    Donne une photo à t=0 de wi(y) et alphai(y) représentative du mode
    Faire une simple valeur abs nous aurait fait perdre les signes ??

    Les faire tourner ne nous fait pas perdre l'info par rapport à la phase entre les vecteurs propres car c'est la phase sur les eta(t) qui nous intéresse
    Puis un v_i tourné est donc orienté de la même manière pour remonter à w_i et alpha_i
    '''

def _stack_mode_shapes(phi_list):
    # Convertit la liste [N_modes] d'array (Ny,) en matrice (N_modes, Ny)
    return np.vstack(phi_list) if len(phi_list) > 0 else np.zeros((0, 1))

def _mode_id_vec(w_mode, alpha_mode):
    '''
    Function to concatenate shapes IN PHYSICAL FIELDS into one vector for correlation with MAC then; use real parts
    '''
    hop = np.hstack([np.asarray(w_mode).ravel(), np.asarray(alpha_mode).ravel()])
    '''
    np.hstack([np.array([1, 2, 3]), np.array([4, 5, 6])])
    -> array([1, 2, 3, 4, 5, 6])
    '''
    return hop

# def _mode_id_from_eigvec(par, vi):
#     '''
#     Function to concatenate the mode shape of A into one vector for correlation with MAC then; use real parts
#     We normalize so the MAC scores can be compared to one an other
#     '''
#     # vi = eigvecs[:par.Nq, k]
#     vw = vi[:par.Nw]
#     va = vi[par.Nw:par.Nq]
#     x = np.concatenate([np.abs(vw), np.abs(va)]).astype(float)
#     n = np.linalg.norm(x) or 1.0 #we might want to test without normalization 
#     return x / n

def _mode_id_from_eigvec(par, vi):
    '''
    Function to concatenate the mode shape of A into one vector for correlation with MAC then; use real parts
    We normalize so the MAC scores can be compared to one an other

    we keep the complex values to keep the phase information
    '''
    vw = vi[:par.Nw]
    va = vi[par.Nw:par.Nq]
    v = np.concatenate([vw, va])
    if v.size:
        k0 = int(np.argmax(np.abs(v)))
        if v[k0] != 0:
            v = v * np.exp(-1j*np.angle(v[k0]))
    n = np.linalg.norm(v) or 1.0
    return v / n

def _rel_phase_from_eigvec(par, vi):
    vw = vi[:par.Nw]
    va = vi[par.Nw:par.Nq]
    if vw.size == 0 or va.size == 0:
        return 0.0
    return float(np.angle(np.vdot(va, vw)))

def _wrap_pi(phi):
    '''
    we want to constrain phi in [-pi, pi]
    useful to compute phase differences

    shortest arc 

    Exemple: prev = 179°, curr = -179°
    Différence brute = -358° ≈ -2π + 2°
    alors qu'en réalité l'écart physique est 2°
    _wrap_pi ramène cette différence à l'intervalle ]−π, π], donc ici ≈ +2°
    '''
    return (phi + np.pi) % (2*np.pi) - np.pi

def _mac(a, b, eps=1e-16):
    '''
    Calculate MAC(a,b)
    1=> same (à une phase près, en fait non ça dépend de la phase relative entre a et b)
    0=> orthogonal
    '''
    num = np.abs(np.vdot(a, b))**2
    den = (np.vdot(a, a).real * np.vdot(b, b).real) + eps
    return num / den

def _assign_by_mac(prev_refs, curr_vecs):
    '''
    On veut maximiser la similarité au sens MAC

    Parameters:
    -----------
    prev_refs: list of K vectors (K=2 here) , Vecteurs suivis à U[i-1]
    curr_vecs: list of N vectors (N=par.Nq) , N vecteurs candidats à U[i]

    Returns:
    -----------
    best[1] : tuple (j0, j1)
        indices des vecteurs qui ressemblent le plus, respectivement au ref prevs_refs
    '''
    K, N = len(prev_refs), len(curr_vecs)
    MAC = np.zeros((K, N))
    '''
    MAC = [ MAC(r1,c1) ... MAC(r1,c_Nq)
            MAC(r2,c1) ... MAC(r2,c_Nq) ]
    '''
    for k in range(K):
        for j in range(N):
            MAC[k, j] = _mac(prev_refs[k], curr_vecs[j]) #we fill the MAX matrix
    # exhaustive assignment for K=2 (robust and simple)
    best_score = -np.inf
    # best_cols = None
    # for j0 in range(N):
    #     for j1 in range(N):
    #         if j1 == j0: # on va regarde le MAC pour toutes les paires possibles (sauf autoMAC ofc)
    #             continue
    #         score = MAC[0, j0] + MAC[1, j1] #on somme le MAC de MAC(w,c1) et MAC(alpha,c1) car le mode c'est la somme des contributions
    #         if score > best[0]:
    #             best = (score, (j0, j1)) #on garde le meilleur
    # return best[1], MAC
    for cols in itertools.permutations(range(N), K):
        score = sum(MAC[k, cols[k]] for k in range(K))
        if score > best_score:
            best_score = score
            best_cols = cols
    return tuple(best_cols), MAC

def _assign_by_mac_with_phase(prev_refs, prev_relphi, curr_vecs, curr_relphi, kappa=0.3):
    '''
    this function is an extension of _assign_by_mac that also takes into account the relative phase between w and alpha for each mode

    Idée: pénaliser les candidats dont la phase relative entre w et alpha diffère de celle du mode suivi à U(i-1)
    kappa in [0,1]: poids de la cohérence de phase
    '''

    K, N = len(prev_refs), len(curr_vecs)
    MAC = np.zeros((K, N)) # MAC matrix
    PHW = np.zeros((K, N)) # phase weight matrix
    for k in range(K):
        for j in range(N):
            MAC[k, j] = _mac(prev_refs[k], curr_vecs[j])
            dphi = _wrap_pi(curr_relphi[j] - prev_relphi[k])
            PHW[k, j] = np.cos(dphi / 2.0) ** 2 # [0,1]
            SCORE = (1 - kappa) * MAC + kappa * (MAC * PHW)
            best_score = -np.inf
            best_cols = None
    for cols in itertools.permutations(range(N), K):
        s = sum(SCORE[k, cols[k]] for k in range(K))
        if s > best_score:
            best_score, best_cols = s, cols
    return tuple(best_cols), MAC, PHW

''' General functions to compute the mode shapes and modal matrices '''

def bendingModeShapes(par):
    """
    Build the bending mode shapes of the wing.

    PARAMETERS
    ---------------
    par : ModelParameters
        Wing parameters (geometry and properties). Expects fields such as
        `Nw`, `has_tip`, `s`, `y`, `m`, and `Mt`.

    RETURNS
    ---------------
    phi_normalized : list[np.ndarray]
        Bending mode shapes φ(y) for each mode (length `Nw`).
    phi_dot_normalized : list[np.ndarray]
        First spatial derivatives dφ/dy for each bending mode.
    phi_dotdot_normalized : list[np.ndarray]
        Second spatial derivatives d²φ/dy² for each bending mode.
    """
    n = par.Nw

    if n < 1:   # Simply check if n is a positive integer
        raise ValueError("n must be a positive integer")

    if par.has_tip is True:     # Case with added mass on the tip, beta values are computed by evaluating the characteristic equation 

        b = np.linspace(0, n*np.pi, n*10000)
        func = 1 + (np.cos(b * par.s) * np.cosh(b * par.s)) + (par.Mt * b/par.airfoil.m) * (np.sinh(b * par.s)*np.cos(b * par.s) - np.sin(b * par.s)*np.cosh(b * par.s))

        func_target = []
        for i in range(len(func)-1):   # To find intersection, we look for a change of sign
            if func[i] * func[i+1] < 0:
                func_target.append(i)

                if len(func_target) == n:   # We stop when we have enough values
                    break

    else:       # Case without added mass on the tip, beta values are computed by evaluating the characteristic equation 

        b = np.linspace(0, n*np.pi, n*10000)
        func = np.cos(b * par.s) * np.cosh(b * par.s) + 1

        func_target = []
        for i in range(len(func)-1):
            if func[i] * func[i+1] < 0: # we seek for a change of sign, that means f(b)=0
                func_target.append(i)

                if len(func_target) == n:
                    break

    # Beta values are stored and used to compute the mode shapes phi(y) and its derivatives
    beta = b[func_target]

    phi_normalized, phi_dot_normalized , phi_dotdot_normalized = [], [], []

    for i in range(n):
        gamma = (np.sin(beta[i] * par.s) + np.sinh(beta[i] * par.s)) / (np.cos(beta[i] * par.s) + np.cosh(beta[i] * par.s))

        a_ = np.sin(beta[i] * par.y) - np.sinh(beta[i] * par.y)
        b_ = np.cos(beta[i] * par.y) - np.cosh(beta[i] * par.y)
        phi = a_ - gamma * b_

        a_ = np.cos(beta[i] * par.y) - np.cosh(beta[i] * par.y)
        b_ = - np.sin(beta[i] * par.y) - np.sinh(beta[i] * par.y)
        phi_dot = (a_ - gamma * b_) * beta[i]


        a_ = np.sin(beta[i] * par.y) + np.sinh(beta[i] * par.y)
        b_ = np.cos(beta[i] * par.y) + np.cosh(beta[i] * par.y)
        phi_dotdot = - (a_ - gamma * b_) * (beta[i]**2)

        modal_mass =  np.trapezoid(phi * par.airfoil.m * phi, par.y) # ∫ m(y) φ(y)² dy

        phi_normalized.append(phi/np.sqrt(modal_mass))#/np.sqrt(modal_mass)
        phi_dot_normalized.append(phi_dot/np.sqrt(modal_mass))#/np.sqrt(modal_mass)
        phi_dotdot_normalized.append(phi_dotdot/np.sqrt(modal_mass))#/np.sqrt(modal_mass)
    
    phi_normalized = np.array(phi_normalized)
    phi_dot_normalized = np.array(phi_dot_normalized)
    phi_dotdot_normalized = np.array(phi_dotdot_normalized)

    # print('w')
    # print(phi_normalized.T.shape)
    return phi_normalized, phi_dot_normalized, phi_dotdot_normalized
    
def torsionModeShapes(par):
    """
    Build the torsion mode shapes of the wing.

    PARAMETERS
    ---------------
    par : object
        Wing parameters (geometry and properties). Expects fields such as
        `Nalpha`, `has_tip`, `s`, `y`, `I_alpha`, and `I_alpha_t`.

    RETURNS
    ---------------
    phi_normalized : list[np.ndarray]
        Torsion mode shapes ψ(y) for each mode (length `Nalpha`).
    phi_dot_normalized : list[np.ndarray]
        First spatial derivatives dψ/dy for each torsion mode.
    phi_dotdot_normalized : list[np.ndarray]
        Second spatial derivatives d²ψ/dy² for each torsion mode.
    """
    n = par.Nalpha

    if n < 1:       # Simply check if n is a positive integer
        raise ValueError("n must be a positive integer")

    if par.has_tip is True:         # Case with added mass on the tip, beta values are computed by evaluating the characteristic equation
        b = np.linspace(0.01, n*np.pi, n*10000)
        func = np.tan(b * par.s) - (par.airfoil.Ialpha_EA * par.s / par.I_alpha_t) * 1/(b * par.s)
        func_target = []

        for i in range(len(func)-1):
            if func[i] * func[i+1] < 0:
                func_target.append(i)

                # Trick : Avoids suspicious intersections by selecting only odd values due to tan() discontinuities at asymptotes
                if len(func_target) == 2*n:   
                    func_target = func_target[::2]
                    break

        beta = b[func_target]

    else:
        beta = []
        for i in range(1,n+1):
            beta.append((2*i-1)*np.pi / (2 * par.s))

    # Beta values are stored and used to compute the mode shapes phi(y) and its derivatives

    phi_normalized, phi_dot_normalized , phi_dotdot_normalized = [], [], []

    for i in range(n):

        phi = np.sin(beta[i] * par.y) 
        
        phi_dot = np.cos(beta[i] * par.y) * beta[i]
        
        phi_dotdot = - np.sin(beta[i] * par.y) * (beta[i] ** 2)

        modal_mass =  np.trapezoid(phi * par.airfoil.Ialpha_EA * phi, par.y)
        phi_normalized.append(phi/np.sqrt(modal_mass))
        phi_dot_normalized.append(phi_dot/np.sqrt(modal_mass))
        phi_dotdot_normalized.append(phi_dotdot/np.sqrt(modal_mass))

    phi_normalized = np.array(phi_normalized)
    phi_dot_normalized = np.array(phi_dot_normalized)
    phi_dotdot_normalized = np.array(phi_dotdot_normalized)

    # print('alphae')
    # print(phi_normalized.T.shape)
    return phi_normalized, phi_dot_normalized, phi_dotdot_normalized

def build_state_q_from_real(par, w_tip=0.0, alpha_tip=0.0, wdot_tip=0.0, alphadot_tip=0.0):
    """
    Construit x0 (2*(Nv+Nw+Nalpha),) à partir de conditions initiales dans les champs
    physiques en bout de poutre: w_tip (m) et alpha_tip (rad), et leurs vitesses.
    Pour ne pas passer pas les q_w et q_alpha qui sont peu intuitifs pour une position initiale par exemple

    Paramètres
    ----------
    par : ModelParameters
        Doit fournir Nw, Nalpha, (optionnellement Nv) et y.
    w_tip : float
        Déflexion en bout (y = s) au temps t=0.
    alpha_tip : float
        Rotation en bout (rad) au temps t=0.
    wdot_tip : float
        Vitesse de déflexion en bout.
    alphadot_tip : float
        Vitesse de rotation en bout.

    Retour
    ------
    x0 : ndarray shape (2*(Nv+Nw+Nalpha),)
        Etat initial [q ; qdot] avec q = [qv, qw, qa].
    """
    import numpy as np

    Nv = par.Nv
    Nw = par.Nw
    Nalpha = par.Nalpha
    n_q = Nv + Nw + Nalpha
    x0 = np.zeros(2 * n_q, dtype=float)

    # Récupère formes modales (discrètes sur par.y), tip = dernier point
    phi_w, _, _ = bendingModeShapes(par)
    phi_a, _, _ = torsionModeShapes(par)
    tip_idx = -1

    # Vecteurs de formes au tip
    if Nw > 0:
        phi_w_tip = np.array([phi_w[i][tip_idx] for i in range(Nw)], dtype=float).reshape(1, -1)
        # phi_w_tip @ qw = w_tip  -> solution min-norme
        '''
        On cherche donc le vecteur qw qui, combiné avec les valeurs des modes au tip, reproduit le déplacement souhaité.
        il n'y pas une solution unique à qw = phi_w_tip^-1 @ w_tip,
        on cherche donc la solution de norme minimal grâce à la fonction np.linalg.lstsq qui utilise la pseudo inverse

        qw = argmin||q||_2 such as Phi_w(tip)q = w(tip)

        Si Nw = 1 alors il y a une seule solution (inverse scalaire classique)

        si Nw > 1 alors infinité de solution

        np.linalg.lstsq rédout Phi_w(tip)@qw = w(tip) en donnant la solution de norme minimale 
        qw = Phi_w[PSEUDO-INVERSE](tip) @ w(tip)

        '''
        qw, *_ = np.linalg.lstsq(phi_w_tip, np.array([w_tip], dtype=float), rcond=None)
        x0[Nv:Nv + Nw] = qw.ravel()
    if Nalpha > 0:
        phi_a_tip = np.array([phi_a[i][tip_idx] for i in range(Nalpha)], dtype=float).reshape(1, -1)
        qa, *_ = np.linalg.lstsq(phi_a_tip, np.array([alpha_tip], dtype=float), rcond=None)
        x0[Nv + Nw:Nv + Nw + Nalpha] = qa.ravel()

    # Vitesses au tip -> qdot via mêmes formes
    if Nw > 0:
        qwdot, *_ = np.linalg.lstsq(phi_w_tip, np.array([wdot_tip], dtype=float), rcond=None)
        x0[n_q + Nv:n_q + Nv + Nw] = qwdot.ravel()
    if Nalpha > 0:
        qadot, *_ = np.linalg.lstsq(phi_a_tip, np.array([alphadot_tip], dtype=float), rcond=None)
        x0[n_q + Nv + Nw:n_q + Nv + Nw + Nalpha] = qadot.ravel()

    return x0

def X_to_q(par, X, t):
    '''
    Function to get the generalised coordinates from the state vector X(t) (X[ qw1 qw2 qw3 qa1 qa2 qa3 + vitesse])
    We just take them from the vector depending on the selected number of modes

    Parameters
    ----------
    par : ModalParameters
    X : np.array [nt*(Nw+Nv+Na)]
        state vector contains the generalised coordinates
    t : 
    Returns
    ----------
    qw_t
    qa_t
        
    '''

    nt = t.size

    # Récupération des coordonnées modales depuis X
    Nv = par.Nv
    Nw = par.Nw
    Na = par.Nalpha
    n_q = X.shape[1] // 2
    Q = X[:, :n_q] # we only take the deplacement coordinates

    i_w0, i_w1 = Nv, Nv + Nw
    i_a0, i_a1 = i_w1, i_w1 + Na
    qw_t = Q[:, i_w0:i_w1] if Nw > 0 else np.zeros((nt, 0))
    qa_t = Q[:, i_a0:i_a1] if Na > 0 else np.zeros((nt, 0))

    return qw_t, qa_t

def _modal_to_physical_fields(par, qw_t, qa_t, return_shapes=False):
    """
    Convertit les coordonnées modales (qw, qa) en champs physiques w(y,t), alpha(y,t).
    EN TEMPOREL
    w(y,t) = Phiw(y) @ qw(t)

    Paramètres
    ----------
    par : ModelParameters
        Doit fournir y, Nw, Nalpha.
    qw_t : ndarray (nt, Nw) ou (Nw,)
        Coordonnées modales en flexion.
    qa_t : ndarray (nt, Nalpha) ou (Nalpha,)
        Coordonnées modales en torsion.
    return_shapes : bool
        Si True, retourne aussi (Phi_w, Phi_alpha).

    Retours
    -------
    w_map : ndarray (nt, Ny)
    alpha_map : ndarray (nt, Ny)
    (optionnel) Phi_w : ndarray (Nw, Ny)
    (optionnel) Phi_alpha : ndarray (Nalpha, Ny)
    """
    import numpy as np

    Nw = par.Nw
    Nalpha = par.Nalpha

    # Garantir 2D (nt, N*)
    qw_t = np.atleast_2d(qw_t)
    qa_t = np.atleast_2d(qa_t)

    # Récupère et empile les formes modales
    phi_w, _, _ = bendingModeShapes(par)      # liste de Nw vecteurs (Ny,)
    phi_a, _, _ = torsionModeShapes(par)      # liste de Nalpha vecteurs (Ny,)

    Ny = len(par.y)
    Phi_w = np.vstack(phi_w)
    Phi_a = np.vstack(phi_a)

    # Champs physiques
    w_map = qw_t @ Phi_w
    alpha_map = qa_t @ Phi_a

    if return_shapes:
        return w_map, alpha_map, Phi_w, Phi_a
    return w_map, alpha_map

def _reconstruct_shapes_from_eigvecs(par, eigvals, eigvecs):
    """
    On reconstruit les contributions de w et alpha pour chaque mode i,
    - en déplacement wi(y) alphai(y)
    - en densité energie cinétique ew(y), ea(y) (normalisée pour pouvoir comparer entre mode car dépend de omega)
    - en dénsité energie elastique Uw(y), Ua(y) (pas besoin de normalisation car ça dépend pas explicitement de omega, en fait si car dans les phidotdot y'a du omega qui apparait dans les Beta)

    les deux types de densité d'énergie donnent pas les mêmes infos

    eigvecs: colonnes = modes (taille 2(Nw+Nalpha) x n_modes) ; on utilise la partie positions.
    Retourne: w_modes, alpha_modes de tailles (n_modes, Ny).
    Phi

    Parameters
    ----------
    par : ModelParameters
    eigvals :
        matrices des valeurs propres déjà ordonnées et triées
    eigvecs:
        matrices des vecteurs propres déjà ordonés et triés

    Returns
    -------
    w_modes, alpha_modes:
        correspondent aux modes Psi_w, càd que c'est les contributions en flexions et torsions SPATIALES des déformées modales du probleme aeroélastique
        c'est wi(y) = Psi_w_i(y) = Phi_w @
    """
    # print("Calcule des contributions des ddl par mode")
    # print(f"normalize = {normalize}")
    Nq = par.Nv + par.Nw + par.Nalpha
    Vq = eigvecs[:Nq, :]  # partie positions, notre vecteur d'état est [qw1 qw2 qw3 qa1 qa2 qa3 qw1' qw2' qw3' qa1' qa2' qa3']'
    nm = Vq.shape[1]

    # energy_type = None
    # if normalize in ('energy', 'energy_mass', 'mass'):
    #     energy_type = 'mass'
    #     Mq = getStructuralMassMatrix(par)
    # elif normalize in ('energy_stiffness', 'stiffness'):
    #     energy_type = 'stiffness'
    #     Kq = getStructuralStiffness(par)

    Mq = getStructuralMassMatrix(par)
    Kq = getStructuralStiffness(par)

    phi_w, phi_dot_w, phi_dotdot_w = bendingModeShapes(par) # phi_w (Nw,Ny) (but not stack)
    phi_alpha, phi_dot_alpha, phi_dotdot_alpha = torsionModeShapes(par) # phi_alpha (Nalpha,Ny) (but not stack)
    Phi_w = _stack_mode_shapes(phi_w)         # (Nw, Ny)
    Phi_alpha = _stack_mode_shapes(phi_alpha) # (Nalpha, Ny)

    Ny = Phi_w.shape[1] if par.Nw > 0 else (Phi_alpha.shape[1] if par.Nalpha > 0 else 0)
    
    # pulsations des modes
    omega = np.imag(eigvals)

    # densité d'énergie cinétique par mode
    Tw = np.zeros((nm, Ny), dtype=float)
    Ta = np.zeros((nm, Ny), dtype=float)

    # énergie totale par mode
    T_Ew = np.zeros((nm, Ny), dtype=float)
    T_Ea = np.zeros((nm, Ny), dtype=float)

    # densité d'énergie cinétique normalisée, nécessaire pour comparer entre les modes car sinon c'est proportionnel à omega²
    T_ew = np.zeros((nm, Ny), dtype=float)
    T_ea = np.zeros((nm, Ny), dtype=float)

    # densité d'énergie élastique 
    Uw = np.zeros((nm, Ny), dtype=float)
    Ua = np.zeros((nm, Ny), dtype=float)
    # énergie totale par mode
    U_Ew = np.zeros((nm, Ny), dtype=float)
    U_Ea = np.zeros((nm, Ny), dtype=float)
    # densité d'énergie normalisée
    U_ew = np.zeros((nm, Ny), dtype=float)
    U_ea = np.zeros((nm, Ny), dtype=float)

    # contribution réelle de w et alpha par mode
    w_modes = np.zeros((nm, Ny), dtype=float)
    alpha_modes = np.zeros((nm, Ny), dtype=float)
    '''
    Ψ_w,i(y) = Φ_w(y) @ vw
    '''

    for i in range(nm): # i parcourt les modes aeroelastiques,
        
        # vi,*_ = _phase_align_column(Vq[:, i]) # qi est comme Vi quand on considère que la forme le eta(t) saute (eta coordonées modales ici)
        vi = Vq[:,i]
        '''
        à partir de ça on peut déjà avoir la part de contribution des modes de déformées de la structure à U=0 dans les modes de déformées du
        système aeroelastique
        '''

        # Normalisation énergétique des vecteurs propres (avant reconstruction des champs)
        '''
        la normalisation des modes par la masse est utile pour comparer les modes entre eux,
        pas utile si on veut simplement comparer alpha et w dans un même mode

        actually as we normalized the spatiales modes shapes of the structure (phi_w phi_a contained in A) we don't
        need to normalize again
        '''
        
        vw = vi[:par.Nw]
        va = vi[par.Nw:Nq]
        # print(vw.shape)
        # print(Phi_w.shape)

        '''
        en fait il faut récupérer w_modes avec la partie imaginaire aussi
        '''

        if par.Nw > 0:
            w_modes_cpx = vw @ Phi_w
            w_dotdot_modes_cpx = vw @ phi_dotdot_w
        if par.Nalpha > 0:
            alpha_modes_cpx = va @ Phi_alpha
            alpha_dot_modes_cpx = va @ phi_dot_alpha

        # densité d'énergie cinétique par mode
        Tw[i,:] = 0.5*par.airfoil.m*omega[i]**2*np.abs(w_modes_cpx)**2 # on prend le module et pas la partie réelle car Re() dépend de la phase du vect propre dont provient wi(y)
        Ta[i,:] = 0.5*par.airfoil.Ialpha_EA*omega[i]**2*np.abs(alpha_modes_cpx)**2 
        # énergie cinétique totale par mode
        T_Ew[i,:] = np.trapezoid(Tw[i,:], par.y)
        T_Ea[i,:] = np.trapezoid(Ta[i,:], par.y)
        # densité énergie cinétique par mode 
        T_ew[i,:] = Tw[i,:]/(T_Ew[i,:]+T_Ea[i,:])
        T_ea[i,:] = Ta[i,:]/(T_Ew[i,:]+T_Ea[i,:])

        Uw[i,:] = 0.5*par.EIx*np.abs(w_dotdot_modes_cpx)**2
        Ua[i,:] = 0.5*par.GJ*np.abs(alpha_dot_modes_cpx)**2
        # énergie cinétique totale par mode
        U_Ew[i,:] = np.trapezoid(Uw[i,:], par.y)
        U_Ea[i,:] = np.trapezoid(Ua[i,:], par.y)
        # densité énergie cinétique par mode 
        U_ew[i,:] = Uw[i,:]/(U_Ew[i,:]+U_Ea[i,:])
        U_ea[i,:] = Ua[i,:]/(U_Ew[i,:]+U_Ea[i,:])

        w_modes[i, :] = np.real(w_modes_cpx)  # (1, Nw) @ (Nw, Ny)
        alpha_modes[i, :] = np.real(alpha_modes_cpx)  # (1, Nalpha) @ (Nalpha, Ny)

    energy_dict = {
    'Tw': Tw,       # densité énergie cinétique brute (w)
    'Ta': Ta,       # densité énergie cinétique brute (alpha)
    'Uw': Uw,       # densité énergie élastique brute (w)
    'Ua': Ua,       # densité énergie élastique brute (alpha)
    'T_Ew': T_Ew,   # énergie cinétique totale (w)
    'T_Ea': T_Ea,   # énergie cinétique totale (alpha)
    'U_Ew': U_Ew,   # énergie élastique totale (w)
    'U_Ea': U_Ea,   # énergie élastique totale (alpha)
    'T_ew': T_ew,   # densité énergie cinétique (w)
    'T_ea': T_ea,   # densité énergie cinétique (alpha)
    'U_ew': U_ew,   # densité énergie élastique (w)
    'U_ea': U_ea,   # densité énergie élastique (alpha)
    }


    '''
    normalize = "per_field" : on prend le max du champ w (resp. a) et on le norm pas son max pour chaque mode
    normalize = "per_mode"  : on prend le max entre a et w pour un mode et ça nous sert de norm

    je pense le mieux c'est de ne pas normaliser comme ça, mais plutôt normaliser a et w par rapport à leur propre energy cinétique,
    comme ça on peut bien comparer a avec w pour chaque mode, et même entre les modes avec un sharey=True (on leur fait partager le même axe)
    '''

    return w_modes, alpha_modes, energy_dict

def modalMatrices(par):
    """
    Build the modal matrices to avoid redundant computations.
    We build the Phi_ww, Phi_alphaalpha and Phi_walpha matrices.

    PARAMETERS
    ---------------
    par : ModelParameters
        Wing parameters and mode shapes settings. Uses `y`, `s`, `Nw`, `Nalpha`.

    RETURNS
    ---------------
    phi_ww : np.ndarray
        Bending–bending modal overlap matrix (shape `Nw x Nw`).
    phi_alphaalpha : np.ndarray
        Torsion–torsion modal overlap matrix (shape `Nalpha x Nalpha`).
    phi_walpha : np.ndarray
        Bending–torsion modal overlap matrix (shape `Nw x Nalpha`).
    """

    phi_w, _, _ = bendingModeShapes(par)
    phi_alpha, _, _ = torsionModeShapes(par)

    # we make the mode shape continuous functions to be able to integrate them
    phi_w_interp = [interp1d(par.y, phi_w[i], kind='cubic', fill_value="extrapolate") for i in range(par.Nw)]
    phi_alpha_interp = [interp1d(par.y, phi_alpha[i], kind='cubic', fill_value="extrapolate") for i in range(par.Nalpha)] #fill_value="extrapolate" to be able to evaluate outside the range of y if needed (or just to correct boundary errors)

    ''' Initialize modal matrices '''
    phi_ww = np.zeros((par.Nw, par.Nw))
    phi_alphaalpha = np.zeros((par.Nalpha, par.Nalpha))
    phi_walpha = np.zeros((par.Nw, par.Nalpha))

    for i in range(par.Nw):
        for j in range(par.Nw):
            def integrand(y):
                return phi_w_interp[i](y) * phi_w_interp[j](y)

            phi_ww[i, j] = quad(integrand, 0, par.s)[0] # phi_ww[i,j] = ∫_0^s φw_i(y) φw_j(y) dy
            '''
            quad is a numerical integrator from scipy that takes a function and the limits of integration
            '''

    for i in range(par.Nalpha):
        for j in range(par.Nalpha):
            def integrand(y):
                return phi_alpha_interp[i](y) * phi_alpha_interp[j](y)

            phi_alphaalpha[i, j] = quad(integrand, 0, par.s)[0]

    for i in range(par.Nw):
        for j in range(par.Nalpha):
            def integrand(y):
                return phi_w_interp[i](y) * phi_alpha_interp[j](y)

            phi_walpha[i, j] = quad(integrand, 0, par.s)[0]

    return phi_ww, phi_alphaalpha, phi_walpha

def getMatrix(phi1,phi2,len1,len2,arg,argtip,par):
    """
    Function used to generalize the computation of the mass and stiffness matrices (structural).
    
    It can only be used for matrices made of element of the 
    form X = int(arg * phi1 * phi2) and Xtip = argtip * phi1[s] * phi2[s] if the tip mass is considered.

    PARAMETERS
    ---------------
    phi1 : list[np.ndarray]
        First set of mode shapes sampled on `par.y` (length `len1`).
    phi2 : list[np.ndarray]
        Second set of mode shapes sampled on `par.y` (length `len2`).
    len1 : int
        Number of modes in `phi1`.
    len2 : int
        Number of modes in `phi2`.
    arg : float or array-like
        Spanwise coefficient inside the integral (e.g., mass per unit length, stiffness per unit length).
    argtip : float
        Tip contribution coefficient (e.g., tip mass or tip inertia).
    par : object
        Wing parameters with fields like `y`, `s`, `has_tip`.

    RETURNS
    ---------------
    X : np.ndarray
        Computed matrix of size (`len1`, `len2`) including tip contributions if enabled.
    """

    X = np.zeros((len1,len2))
    Xtip = np.zeros((len1,len2)) 

    phi1_interp = [interp1d(par.y, phi1[i], kind='cubic', fill_value="extrapolate") for i in range(len1)]
    phi2_interp = [interp1d(par.y, phi2[j], kind='cubic', fill_value="extrapolate") for j in range(len2)]

    for i in range(len1):
        for j in range(len2):
            def integrand(y):
                return arg * phi1_interp[i](y) * phi2_interp[j](y)

            X[i, j] = quad(integrand, 0, par.s)[0] # X[i,j] = ∫_0^s arg * phi1_i(y) * phi2_j(y) dy
            '''
            arg can be either mass per unit length, bending stiffness per unit length, torsional inertia per unit length or torsional stiffness per unit length
            '''

            if par.has_tip is True:
                Xtip[i, j] = argtip * phi1[i][-1] * phi2[j][-1]

    return X + Xtip

''' Computation of the structural matrices '''

def getStructuralMassMatrix(par):
    """
    Compute the structural mass matrix of the wing.

    PARAMETERS
    ---------------
    par : object
        Wing parameters and mode counts.

    RETURNS
    ---------------
    M : np.ndarray
        Structural mass matrix of shape (`Nw+Nalpha`, `Nw+Nalpha`).
    """

    phi_w, _, _ = bendingModeShapes(par)
    phi_alpha, _, _ = torsionModeShapes(par)

    Mww = getMatrix(phi_w , phi_w , par.Nw , par.Nw , par.airfoil.m , par.Mt , par)
    Mwalpha = getMatrix(phi_w , phi_alpha , par.Nw , par.Nalpha , par.airfoil.m*par.airfoil.x_alpha , par.Mt*par.x_t , par)
    Malphaalpha = getMatrix(phi_alpha , phi_alpha , par.Nalpha , par.Nalpha , par.airfoil.Ialpha_EA , par.I_alpha_t , par)

    M = np.block([[Mww, Mwalpha], [ Mwalpha.T, Malphaalpha]])

    return M
    
def getStructuralStiffness(par):
    """
    Compute the structural stiffness matrix of the wing.

    PARAMETERS
    ---------------
    par : object
        Wing parameters and mode counts.

    RETURNS
    ---------------
    K : np.ndarray
        Structural stiffness matrix of shape (`Nw+Nalpha`, `Nw+Nalpha`).
    """

    _, _, phi_dotdot_w = bendingModeShapes(par)
    _, phi_dot_alpha, _ = torsionModeShapes(par)
    
    Kww = getMatrix(phi_dotdot_w , phi_dotdot_w , par.Nw , par.Nw , par.EIx , 0, par)
    Kalphaalpha = getMatrix(phi_dot_alpha , phi_dot_alpha , par.Nalpha , par.Nalpha , par.GJ , 0, par)
    
    K = np.block([[Kww, np.zeros((par.Nw,par.Nalpha))], [np.zeros((par.Nalpha,par.Nw)), Kalphaalpha]])

    return K

def getStructuralDamping(par):
    """
    Compute the structural damping matrix of the wing.

    PARAMETERS
    ---------------
    par : object
        Wing parameters and mode counts (includes modal damping ratios).

    RETURNS
    ---------------
    C : np.ndarray
        Structural damping matrix of shape (`Nw+Nalpha`, `Nw+Nalpha`).
    """

    M = getStructuralMassMatrix(par)
    K = getStructuralStiffness(par)

    Mww = M[:par.Nw,:par.Nw]
    Malphaalpha = M[par.Nw:,par.Nw:]
    Kww = K[:par.Nw,:par.Nw]
    Kalphaalpha = K[par.Nw:,par.Nw:]

    Cww = np.zeros((par.Nw,par.Nw))
    Calphaalpha = np.zeros((par.Nalpha,par.Nalpha))

    for i in range(par.Nw):
        Cww[i,i] = 2 * par.eta_w * np.sqrt(Kww[i,i] * Mww[i,i])
    for j in range(par.Nalpha):
        Calphaalpha[j,j] = 2 * par.eta_alpha * np.sqrt(Kalphaalpha[j,j] * Malphaalpha[j,j])

    C = np.block([[Cww, np.zeros((par.Nw , par.Nalpha))], [np.zeros((par.Nalpha , par.Nw)), Calphaalpha]])

    return C

''' Main function to extract the structural matrices '''

def StructuralMatrices(model_params):
    """
    Compute the structural matrices of the wing.

    PARAMETERS
    ---------------
    model_params : object
        Model parameters container passed to internal routines.

    RETURNS
    ---------------
    M : np.ndarray
        Structural mass matrix.
    C : np.ndarray
        Structural damping matrix.
    K : np.ndarray
        Structural stiffness matrix.
    """

    par = model_params

    M = getStructuralMassMatrix(par)
    K = getStructuralStiffness(par)
    C = getStructuralDamping(par)

    return M, C, K

''' Computation of aerodynamic matrices '''

def QuasiSteadyAeroModel(par,U):
    """
    Build the added aerodynamic stiffness matrix predicted by the quasi-steady model.

    PARAMETERS
    ---------------
    par : object
        Wing and aerodynamic parameters (uses `rho_air`, `c`, `dCn`, `dCm`, mode counts).
    U : float
        Freestream velocity (m/s).

    RETURNS
    ---------------
    Ka : np.ndarray
        Added aerodynamic stiffness matrix of shape (`Nw+Nalpha`, `Nw+Nalpha`).
    """

    phi_ww, phi_alphaalpha, phi_walpha = modalMatrices(par)
    Ka = 0.5 * par.rho_air * (U**2) * par.airfoil.c * np.block([[0 * phi_ww , par.dCn * phi_walpha], [0 * phi_walpha.T , - par.airfoil.c * par.dCm * phi_alphaalpha]])
  
    return Ka 

def TheodorsenFunction(k):
    """
    Compute the Theodorsen function for a given reduced frequency.

    PARAMETERS
    ---------------
    k : float
        Reduced frequency (k = ω b / U).

    RETURNS
    ---------------
    C_real : float
        Real part of the Theodorsen function C(k).
    C_imag : float
        Imaginary part of the Theodorsen function C(k).
    """

    # if k <= 1/2 :
    #     C =  1 - ((0.165)/(1 - 0.0455j/k)) - ((0.335)/(1 - 0.3j/k))

    # else :
    #     C =  1 - ((0.165)/(1 - 0.041j/k)) - ((0.335)/(1 - 0.32j/k))

    # return np.real(C), np.imag(C)

    K0 = kv(0, 1j * k) #kv is the modified Bessel function of the second kind from scipy.special
    K1 = kv(1, 1j * k)

    C = K1 / (K0 + K1)

    return np.real(C), np.imag(C)

def CooperWrightNotation(par,k):
    """
    Convert the Theodorsen function into the Cooper-Wright notation.

    PARAMETERS
    ---------------
    par : object
        Aerodynamic parameters (uses `dCn`, `dCm`, `a`).
    k : float
        Reduced frequency (k = ω b / U).

    RETURNS
    ---------------
    L_w_dot_dot : float
        Added-lift coefficient for w¨ term.
    L_w_dot : float
        Added-lift coefficient for w˙ term.
    L_w : float
        Added-lift coefficient for w term.
    L_alpha_dot_dot : float
        Added-lift coefficient for α¨ term.
    L_alpha_dot : float
        Added-lift coefficient for α˙ term.
    L_alpha : float
        Added-lift coefficient for α term.
    M_w_dot_dot : float
        Added-moment coefficient for w¨ term.
    M_w_dot : float
    Added-moment coefficient for w˙ term.
    M_w : float
        Added-moment coefficient for w term.
    M_alpha_dot_dot : float
        Added-moment coefficient for α¨ term.
    M_alpha_dot : float
        Added-moment coefficient for α˙ term.
    M_alpha : float
        Added-moment coefficient for α term.
    """

    F,G = TheodorsenFunction(k)

    L_w_dot_dot = np.pi
    L_w_dot = par.dCn * F
    L_w = - par.dCn * k * G

    L_alpha_dot_dot = - np.pi * par.airfoil.a
    L_alpha_dot = par.dCn * ( F * (1/2 - par.airfoil.a) + (G/k)) + np.pi
    L_alpha = par.dCn * (F - k * G * (1/2 - par.airfoil.a) )

    M_w_dot_dot = np.pi * par.airfoil.a
    M_w_dot = 2 * par.dCm * F
    M_w = - 2 * par.dCm * k * G 

    M_alpha_dot_dot = - np.pi * (1/8 + (par.airfoil.a**2))
    M_alpha_dot = 2 * par.dCm * (F * (1/2 - par.airfoil.a) + (G/k)) + (-np.pi * (1/2 - par.airfoil.a))
    M_alpha =  2 * par.dCm * (F - k * G * (1/2 - par.airfoil.a))

    return L_w_dot_dot, L_w_dot, L_w, L_alpha_dot_dot, L_alpha_dot, L_alpha, M_w_dot_dot, M_w_dot, M_w, M_alpha_dot_dot, M_alpha_dot, M_alpha

def TheodoresenAeroModel(par,U,omega):
    """
    Build the added aerodynamic mass, stiffness and damping matrices predicted by the Theodoresen model.
    """
    if U!=0:
        k = par.airfoil.b * omega / U
        L_w_dot_dot, L_w_dot, L_w, L_alpha_dot_dot, L_alpha_dot, L_alpha, M_w_dot_dot, M_w_dot, M_w, M_alpha_dot_dot, M_alpha_dot, M_alpha = CooperWrightNotation(par,k)
        phi_ww, phi_alphaalpha, phi_walpha = modalMatrices(par)

        Ka = par.rho_air * (U**2) * np.block([[L_w * phi_ww , L_alpha * par.airfoil.b * phi_walpha], [- M_w * par.airfoil.b * phi_walpha.T, - M_alpha * (par.airfoil.b**2) * phi_alphaalpha]])
        Ca = par.rho_air * U * par.airfoil.b *np.block([[L_w_dot * phi_ww , L_alpha_dot * par.airfoil.b * phi_walpha], [- M_w_dot * par.airfoil.b * phi_walpha.T, - M_alpha_dot * (par.airfoil.b**2) * phi_alphaalpha]])
        Ma = par.rho_air * (par.airfoil.b**2) * np.block([[L_w_dot_dot * phi_ww , L_alpha_dot_dot * par.airfoil.b * phi_walpha], [- M_w_dot_dot * par.airfoil.b * phi_walpha.T, - M_alpha_dot_dot * (par.airfoil.b**2) * phi_alphaalpha]])
        # Ma does not depend on U
    else:
        Ka = np.zeros((par.Nalpha+par.Nw+par.Nv,par.Nalpha+par.Nw+par.Nv))
        Ca = np.zeros((par.Nalpha+par.Nw+par.Nv,par.Nalpha+par.Nw+par.Nv))
        Ma = np.zeros((par.Nalpha+par.Nw+par.Nv,par.Nalpha+par.Nw+par.Nv))
    return Ka, Ca, Ma

''' Eigenvalue problem and modal parameters extraction '''

''' At rest '''

def ModalParamAtRest(par):
    """
    Compute natural frequencies (at rest, no aerodynamic coupling).

    PARAMETERS
    ---------------
    par : ModelParameters
        Wing parameters and mode counts.
    normalize : "per_mode", or "per_field"

    RETURNS
    ---------------
    f0 : np.ndarray
        Natural frequencies in Hz for the structural modes.
    """

    M = getStructuralMassMatrix(par)
    K = getStructuralStiffness(par)
    C = getStructuralDamping(par)

    Minv = np.linalg.inv(M)

    A = np.block([
        [np.zeros((par.Nw + par.Nalpha, par.Nw + par.Nalpha)), np.eye(par.Nw + par.Nalpha)],
        [-Minv @ K, -Minv @ C]
    ])

    eigvals, eigvecs = _eigs_sorted_positive_imag(A)
    freqs = np.imag(eigvals) / (2 * np.pi)
    zeta = -np.real(eigvals) / np.abs(eigvals) # zeta = -p/sqrt(p²+wn²)

    w_modes, alpha_modes, energy_dict = _reconstruct_shapes_from_eigvecs(par, eigvals, eigvecs)

    return freqs, zeta, eigvals, eigvecs, w_modes, alpha_modes, energy_dict

''' Dynamic '''

def stateMatrixAero(par,U,omega):
    """
    Build the state matrix of the system (structural and aerodynamic consideration).

    PARAMETERS
    ---------------
    par : ModelParameters from input.py
        Wing and aerodynamic parameters, including model selection `model_aero`.
    U : float
        Freestream velocity (m/s).
    omega : float
        Circular frequency used in unsteady aerodynamics (rad/s).

    RETURNS
    ---------------
    A : np.ndarray
        State matrix of size (2(Nw+Nalpha) x 2(Nw+Nalpha)).
    """

    M = getStructuralMassMatrix(par)
    C = getStructuralDamping(par)
    K = getStructuralStiffness(par)

    if par.model_aero == 'QuasiSteady':
        Ka = QuasiSteadyAeroModel(par,U)
        K = K + Ka
        Minv = np.linalg.inv(M)

    elif par.model_aero == 'Theodorsen':
        Ka , Ca , Ma = TheodoresenAeroModel(par,U,omega)
        K = K + Ka
        C = C + Ca
        Minv = np.linalg.inv(M+Ma)
    else:
        raise ValueError("Model not recognized")
    
        
    A = np.block([[np.zeros((par.Nw + par.Nalpha,par.Nw + par.Nalpha)),np.eye(par.Nw + par.Nalpha)],[-Minv @ K,-Minv @ C]])
        
    return A

def ModalParamDyn(par, tracked_idx=(0,1,2,3), compute_shapes=False, compute_energy=False, track_using = None):
    '''
    Compute the modal parameters of the system for a range of wind speeds.

    PARAMETERS
    ---------------
    par : ModelParameters
        Parameters including `U` (array of wind speeds) and structural/aero settings.
    tracked_idx : list
        the modes that we track

    RETURNS
    ---------------
    f : np.ndarray
        Modal frequencies in Hz for two tracked modes (shape len(U) x 2).
    damping : np.ndarray
        Modal damping ratios for the two modes (shape len(U) x 2).
    w_modes, alpha_modes :
        equivalent to Psi_wi Psi_ai, spatial modal deformations of the aeroelastic model (not in time!!!)
        we recall that w_i(y) = Psi_wi = Phi_w @ V (@ eta(t)     w/ eta(t) not considered)
    '''

    U = par.U
    # Pour Theodorsen, nécessite un omega de ref pour calculer la frequence réduite pour construire A
    freqs_struc, *_ = ModalParamAtRest(par)
    omega_struc = 2*np.pi*freqs_struc


    # choose initial tracked modes indices and set references
    # tracked_idx = (1, 2)  # we follow the B2 and T1 mode
    # tracked_idx = (0,1,2,3)  # actually we want to follow more than 2 modes

    f = np.zeros((len(U),len(tracked_idx)))
    damping = np.zeros((len(U),len(tracked_idx)))
    realpart = np.zeros((len(U),len(tracked_idx)))

    idx0 = [j for j in tracked_idx if j < len(omega_struc)]
    previous_omega = float(np.mean(omega_struc[idx0])) if idx0 else float(np.mean(omega_struc))

    #previous_omega = (omega_struc[1] + omega_struc[2])/2 # must be changed when we'll consider the v-DOF
    prev = []

    # we keep the contributions of each DDL w, alpha for each U
    # maybe we can only follow the modes [1] and [2], not the [0] and the others
    Ny = len(par.y)

    
    w_modes_U = np.zeros((len(U), par.Nq, Ny))
    alpha_modes_U = np.zeros((len(U), par.Nq, Ny))
    f_modes_U = np.zeros((len(U), par.Nq)) # not the same as f, this is the frequency of each mode at each U, not only the tracked ones

    # densité énergie cinétique normalisée car ça dépend de omega² sinon
    T_ew_U = np.zeros((len(U), par.Nq, Ny))
    T_ea_U = np.zeros((len(U), par.Nq, Ny))

    # densité énergie élastique
    U_ew_U = np.zeros((len(U), par.Nq, Ny))
    U_ea_U = np.zeros((len(U), par.Nq, Ny))

    # Stocke les vecteurs propres complexes sans perte de la partie imaginaire
    eigvecs_U = np.zeros((len(U), 2*par.Nq, par.Nq), dtype=complex)  # matrice pour stocker les vecteurs propres de A

    if (track_using == 'fields'):
        compute_shapes = True  # we need shapes to track using fields

    for i in range(len(U)): # i parcourt les différentes vitesses
        prev.append(previous_omega)
        
        A = stateMatrixAero(par,U[i],previous_omega)
        '''
        big problem here, we can't order the frequencies like that, when T1 and B2 crosses it will mess everythg,
        SOLVED with mode tracking with MAC
        '''
        
        eigvals, eigvecs = _eigs_sorted_positive_imag(A)
        

        w = np.imag(eigvals)
        f0 = w/(2*np.pi)

        eigvecs_U[i,:,:] = eigvecs
        f_modes_U[i, :] = f0
        '''
        actually, we must reorder the modes to keep tracking the same modes even in this eigvecs_U and f_modes_U matrices
        it works to keep the brut order if the modes don't cross, but when they cross it fails
        the tracking with MAC must be implemented for all the modes and not only for the followed ones
        '''

        p = np.real(eigvals)
        zeta = -np.real(eigvals) / np.abs(eigvals) # = - real(lambda) / abs(lambda) = - real(lambda) / sqrt(real(lambda)^2 + imag(lambda)^2)

        '''necessary for the following modes'''
        # only if shapes/energy requested
        if compute_shapes or compute_energy:
            w_modes, alpha_modes, energy_dict = _reconstruct_shapes_from_eigvecs(par, eigvals, eigvecs)
            if compute_shapes:
                w_modes_U[i, :, :] = w_modes
                alpha_modes_U[i, :, :] = alpha_modes
            if compute_energy:
                T_ew_U[i, :, :] = energy_dict['T_ew']
                T_ea_U[i, :, :] = energy_dict['T_ea']
                U_ew_U[i, :, :] = energy_dict['U_ew']
                U_ea_U[i, :, :] = energy_dict['U_ea']

        # modes index must be changed when we'll add the v-DOF
        # we only keep the 2nd [1] and 3rd [2] mode (usually it's the 2nd bending mode and 1st torsion mode)
        
        # mode ID vectors for tracking (either 'field' or None)
        # field : we use the w_modes and a_modes shapes in the physical space to track the modes
        # None  : we use the eigvecs vi from A to track the modes
        if i == 0:
            curr_vecs = [_mode_id_from_eigvec(par, eigvecs[:par.Nq, j]) for j in range(par.Nq)] #we have a j loop to consider all the modes
            curr_rel = [_rel_phase_from_eigvec(par, eigvecs[:par.Nq, j]) for j in range(par.Nq)]
            prev_refs = [curr_vecs[j] for j in tracked_idx]
            prev_rel = [curr_rel[j] for j in tracked_idx] # on regarde aussi la phase relative entre vw et va Vq1 = [vw ; va]
            # mode à U=0 devient notre ref pour la suite
        else:
            # assign by MAC against previous references
            curr_vecs = [_mode_id_from_eigvec(par, eigvecs[:par.Nq, j]) for j in range(par.Nq)]
            curr_rel = [_rel_phase_from_eigvec(par, eigvecs[:par.Nq, j]) for j in range(par.Nq)]

            tracked_idx, MAC, PHW = _assign_by_mac_with_phase(prev_refs, prev_rel, curr_vecs, curr_rel, kappa=0.3)
            # _assign_by_mac_with_phase returns the new indices of the tracked modes considering both MAC and relative phase criteria
            # _assign_by_mac only uses MAC for the assignment

            # update references
            prev_refs = [curr_vecs[j] for j in tracked_idx]
            prev_rel = [curr_rel[j] for j in tracked_idx]

        # fill outputs for the tracked modes only
        k_list = tracked_idx
        for j, k in enumerate(k_list):
            f[i, j] = f0[k]
            damping[i, j] = zeta[k]
            realpart[i, j] = p[k]

        # keep omega ref near the followed modes (updating the reduced frequency for Theodorsen model)
        # previous_omega = float(np.mean(w[list(tracked_idx)]))
        previous_omega = float(w[k_list[0]]+w[k_list[1]])/2
        # JE SAIS pas si on doit faire la moyenne de tous les omega pour la freq réduite ou que des modes qui nous intéresse ?
        # previous_omega = 0.5 * (w[k_list[1]] + w[k_list[]])
    
    energy_dict_U = {
        'T_ew_U' : T_ew_U,
        'T_ea_U' : T_ea_U,
        'U_ew_U' : U_ew_U,
        'U_ea_U' : U_ea_U
    }

    return f, damping , eigvecs_U, f_modes_U, w_modes_U, alpha_modes_U, energy_dict_U

''' Objectives computations'''

def obj_evaluation(U, damping, return_status=False):
    '''
    Function to evaluate when the damping (of the Torsion mode) crosses 0,
    we also evaluate the slope when it crosses (if it crosses, otherwise near it)
    
    In flutter case we always send the TORSION mode damping(U), 3rd mode of the struc
    when balancement is not considered (so usually 2nd column of damping from ROM evaluation)

    Parameters
    ----------
    U : np.array
        list of the different used U values
    damping : np.array
    '''
    # damp_2d: shape (nU, n_modes)
    Uc_best = None
    slope_cross = None

    Umax = U[-1]
    Uc_malus = Umax  # penalization value if no crossing found
    # ce malus n'a du sens que si on empêche l'extrapolation, sinon
    # ça voudrait dire qu'on peut avoir un meilleur score avec le malus qu'avec une extrapolation de pente negative

    beta = 20 # Uc = Uc_malus + beta * damping[-1]

    d = np.asarray(damping, dtype=float)
    s = np.gradient(d, U)

    # look for the first possitive to negative crossing >0 -> <=0
    cross = np.where((d[:-1] > 0.0) & (d[1:] <= 0.0))[0]

    # If a crossing exists :
    if cross.size > 0:
        j = int(cross[0])
        ''' # linear interpolation to have a precise Uc
        # du = U[j+1] - U[j]
        # dd = d[j+1] - d[j]
        # if dd != 0:
        #     alpha = -d[j] / dd
        #     Uc = U[j] + alpha * du
        # else:
        #     Uc = U[j]
        # slope = dd / du if du != 0 else 0.0

        # Uc_best = Uc
        # slope_best = slope_best
        '''

        Uc_best = U[j]
        slope_cross = s[j]
        status = 'cross'
    else:  # if the damping doesn't cross 0
        '''
        # Require at least two consecutive negative slope steps to allow extrapolation
        # it avoids the problem when the MAC tracking messes up the damping curve and we have some isolated negative slopes (specially at low U)
        def at_least_k_consecutive_true(mask: np.ndarray, k: int) -> np.ndarray:
            # renvoie les indices de départ des séquences de True d'au moins k
            if k <= 1:
                return np.where(mask)[0]
            starts = mask.copy()
            for shift in range(1, k):
                starts = starts[:-1] & mask[shift:]
                mask = mask  # (juste pour clarté, pas nécessaire)
            return np.where(starts)[0]
        k = 3
        neg = s < 0.0
        starts_k = at_least_k_consecutive_true(neg, k)
        if starts_k.size > 0:
            # candidate indices are the union of starts and starts+1
            cand_idx = np.unique(np.concatenate([starts_k + t for t in range(k)]))

            # pick the most negative slope among candidates
            j = cand_idx[np.argmin(s[cand_idx])]
            
            if U[j] > 8 : # to extrapolate only for realistic Uc
                Uc_hat = U[j] - d[j] / s[j]
                Uc_best = Uc_hat
                slope_cross = s[j]
                status = 'extrapolated'
                
            else :
                # unrealistic extrapolated Uc : censor
                Uc_best = 70
                slope_cross = 0.0
                status = 'censored'
            
        else:
        '''
            # not enough evidence (no 2 consecutive negative slopes): censor
        count = 0
        count = sum(s[i] * s[i-1] < 0 for i in range(1, len(s)))

        if count ==0 : # no negative slope at all
            Uc_best = Uc_malus + beta*damping[-1]  # penalization value if no crossing found
            slope_cross = 0.0
            status = 'censored'
        else : # that means we are dealing with a hump mode and we try to avoid them (because to hard to deal with them to make them cross)
            Uc_best = Uc_malus + beta + beta/10*damping[-1]
            '''
            penalization value if no crossing found
            commence ça on est sur de pas avoir un meilleur score pour un hump mode qu'avec un potentiel vrai crossing hors fenetre
            car damping in [0,1] quand ça cross pas
            '''
            slope_cross = 0.0
            status = 'censored'




    if return_status:
        return Uc_best, slope_cross, status
    else:
        return Uc_best, slope_cross


''' Temporal resolutions '''

def integrate_state_rk(par, U, t, x0=None, omega_ref=None, return_A=True, rk_order = 2):
    """
    Intègre temporellement x' = A x avec A = stateMatrixAero(par, U, omega_ref)
    en Runge-Kutta ordre 1 (Euler explicite).

    Paramètres
    ----------
    par : ModelParameters
        Doit être compatible avec stateMatrixAero/ModalParamAtRest.
    U : float
        Vitesse d'écoulement choisie (m/s).
    t : array-like (n_t,)
        Grille de temps croissante.
    x0 : array-like (n_x,), optionnel
        Etat initial. Par défaut vecteur nul.
    omega_ref : float, optionnel
        w de ref utilisée par Theodorsen pour frequence réduite k 
        pour construire A. Si None, on la déduit des fréquences propres
        au repos comme 0.5*(omega_2 + omega_3) quand disponible.
    return_A : bool
        Si True, retourne aussi la matrice d'état utilisée.

    Retours
    -------
    t : ndarray (n_t,)
    X : ndarray (n_t, n_x)
        Evolution temporelle de l'état.
    A : ndarray (n_x, n_x) si return_A==True
        Matrice d'état utilisée pour l'intégration.

    Remarques
    ---------
    - Méthode d'ordre 1 (Euler explicite). Choisir un pas de temps suffisamment petit.
    - Pour par.model_aero == 'Theodorsen', U doit être > 0.
    """

    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t should be a 1D vector.")

    # Dimension d'état: 2(Nw+Nalpha)
    n_q = int(par.Nv + par.Nw + par.Nalpha)
    n_x = 2 * n_q

    # Etat initial
    if x0 is None:
        x = np.zeros(n_x, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).ravel()
        if x.size != n_x:
            raise ValueError(f"x0 should be a vector : {n_x}, not : {x.size}.")

    # Fréquence de référence (Theodorsen)
    if par.model_aero == "Theodorsen":
        if U < 0:
            raise ValueError("Theodorsen needs U >= 0")
        if omega_ref is None:
            # on calcule une frequence de ref à partir du modèle au repos
            freqs_struc, *_ = ModalParamAtRest(par)  # Hz
            omega_struc = 2 * np.pi * np.asarray(freqs_struc, dtype=float)
            if omega_struc.size >= 3:
                omega_ref = 0.5 * (omega_struc[1] + omega_struc[2])
            elif omega_struc.size >= 1:
                omega_ref = float(omega_struc[-1])
            else:
                omega_ref = 0.0
    else:
        # Modèles quasi-statiques n'utilisent pas omega_ref
        if omega_ref is None:
            omega_ref = 0.0

    # Matrice d'état (constante pendant l'intégration)
    A = stateMatrixAero(par, U, omega_ref)

    # Intégration Euler explicite
    nt = t.size
    X = np.zeros((nt, n_x), dtype=float)
    X[0, :] = x

    if rk_order not in (1, 2, 3, 4):
        raise ValueError("rk_order must be in {1,2,3,4}")

    for k in range(nt - 1):
        dt = t[k + 1] - t[k]
        if dt <= 0:
            raise ValueError("dt must be >0")

        if rk_order == 1:
            # Euler explicite (RK1): x_{k+1} = x_k + dt * A x_k
            k1 = A @ x
            x = x + dt * k1

        elif rk_order == 2:
            # RK2 (midpoint)
            k1 = A @ x
            k2 = A @ (x + 0.5 * dt * k1)
            x = x + dt * k2

        elif rk_order == 3:
            # RK3 (Kutta 3 classique)
            k1 = A @ x
            k2 = A @ (x + 0.5 * dt * k1)
            k3 = A @ (x + dt * (-k1 + 2.0 * k2))
            x = x + (dt / 6.0) * (k1 + 4.0 * k2 + k3)

        else:  # rk_order == 4
            # RK4 classique
            k1 = A @ x
            k2 = A @ (x + 0.5 * dt * k1)
            k3 = A @ (x + 0.5 * dt * k2)
            k4 = A @ (x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        X[k + 1, :] = x

    if return_A:
        return t, X, A
    return t, X

''' Plot functions '''

def plot_w_alpha_fields(par, t, X, U = None, times_to_plot=None, cmap='viridis', return_maps=False):
    """
    Reconstruit et trace w(y,t) et alpha(y,t) à partir de l'état X(t).
    Suppose q = [qv (optionnel), qw (Nw), qa (Nalpha)] puis [vitesse...].

    Paramètres
    ----------
    par : ModelParameters
        Doit fournir y, Nw, Nalpha, (optionnellement Nv).
    t : ndarray (nt,)
        Temps utilisés lors de l'intégration.
    X : ndarray (nt, 2*(Nv+Nw+Nalpha))
        Trajectoires d'état issues de l'intégration temporelle
    times_to_plot : list[float] ou None
        Instants sélectionnés pour coupes w(y, t_i) et alpha(y, t_i). Si None, choisi 4 instants.
    cmap : str
        Colormap pour les cartes temps-envergure.
    return_maps : bool
        Si True, retourne aussi (w_map, alpha_map) de taille (nt, Ny).

    Retours
    -------
    (optionnel) w_map : ndarray (nt, Ny), alpha_map : ndarray (nt, Ny)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.asarray(t, dtype=float)
    nt = t.size
    if X.shape[0] != nt:
        raise ValueError("X and t must have the same number of rows.")

    # Récupère formes modales

    Nw = par.Nw
    Nalpha = par.Nalpha
    Nv = par.Nv

    # Slices des coordonnées modales dans q
    n_q = X.shape[1] // 2
    Q = X[:, :n_q]  # (nt, Nv+Nw+Nalpha)
    i_w0 = Nv
    i_w1 = Nv + Nw
    i_a0 = i_w1
    i_a1 = i_w1 + Nalpha

    qw_t = Q[:, i_w0:i_w1]
    qa_t = Q[:, i_a0:i_a1]

    # Mapping modal -> champs
    w_map, a_map = _modal_to_physical_fields(par, qw_t, qa_t)

    # Choix des instants pour coupes
    if times_to_plot is None:
        # 4 instants répartis (début/extrémités incluses)
        idx = np.linspace(0, nt - 1, 4, dtype=int)
    else:
        times_to_plot = np.asarray(times_to_plot, dtype=float)
        # projection aux indices les plus proches
        idx = np.array([np.argmin(np.abs(t - ti)) for ti in times_to_plot], dtype=int)

    # Figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    if U>=0: # si U = None alors ça renvoie False par défaut
        fig.suptitle(f'U = {U} m/s')
    # Cartes temps–envergure
    im0 = axes[0, 0].imshow(w_map, aspect='auto', origin='lower',
                            extent=[par.y[0], par.y[-1], t[0], t[-1]], cmap=cmap)
    axes[0, 0].set_title("w(y,t)")
    axes[0, 0].set_xlabel("y")
    axes[0, 0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0, 0], label='w')

    im1 = axes[1, 0].imshow(a_map, aspect='auto', origin='lower',
                            extent=[par.y[0], par.y[-1], t[0], t[-1]], cmap=cmap)
    axes[1, 0].set_title("alpha(y,t)")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1, 0], label='alpha')

    # Coupes w(y, t_i)
    for j in idx:
        axes[0, 1].plot(par.y, w_map[j, :], label=f"t={t[j]:.3g}")
    axes[0, 1].set_title("Coupes w(y, t_i)")
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("w")
    axes[0, 1].legend(loc='best')

    # Coupes alpha(y, t_i)
    for j in idx:
        axes[1, 1].plot(par.y, a_map[j, :], label=f"t={t[j]:.3g}")
    axes[1, 1].set_title("Coupes alpha(y, t_i)")
    axes[1, 1].set_xlabel("y")
    axes[1, 1].set_ylabel("alpha")
    axes[1, 1].legend(loc='best')

    plt.show()

    if return_maps:
        return w_map, a_map
    
def plot_tip_time_and_fft(par, t, X, detrend=True, U=None, window=True, zero_pad=1, freq_max=None, return_data=False):
    """
    Trace w(y=s,t) et alpha(y=s,t) + leurs FFT (Hz) en disposition 2x2:
        w(y=s,t)        |  FFT(w(y=s,t))
        alpha(y=s,t)    |  FFT(alpha(y=s,t))

    Paramètres
    ----------
    par : ModelParameters
        Doit fournir y, Nw, Nalpha, (optionnellement Nv).
    t : ndarray (nt,)
        Temps utilisés lors de l'intégration (pas uniforme recommandé).
    X : ndarray (nt, 2*(Nv+Nw+Nalpha))
        Trajectoires d'état issues de l’intégration.
    detrend : bool
        Si True, retire la moyenne avant FFT.
    window : bool
        Si True, applique une fenêtre de Hann avant FFT.
    zero_pad : int
        Facteur de zero-padding (>=1).
    freq_max : float ou None
        Limite max sur l’axe des fréquences (Hz). None = Nyquist.
    return_data : bool
        Si True, retourne aussi (t, w_tip, alpha_tip, f_w, W_mag, f_a, A_mag).

    Retours (optionnels si return_data=True)
    ---------------------------------------
    t : ndarray (nt,)
    w_tip : ndarray (nt,)
    alpha_tip : ndarray (nt,)
    f_w : ndarray (nf,)
    W_mag : ndarray (nf,)
    f_a : ndarray (nf,)
    A_mag : ndarray (nf,)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Sécurité basique
    t = np.asarray(t, dtype=float)
    nt = t.size
    if X.shape[0] != nt:
        raise ValueError("X doit avoir autant de lignes que t d'éléments.")
    if nt < 2:
        raise ValueError("Besoin d'au moins 2 points temporels.")
    dt = np.mean(np.diff(t))
    if not np.allclose(np.diff(t), dt, rtol=1e-3, atol=1e-9):
        raise ValueError("t doit être (quasi) uniformément échantillonné pour la FFT.")

    # we get the generalized coordinates from the state vector X(t)
    qw_t, qa_t = X_to_q(par = par, X=X, t=t)

    # Mapping modal -> champs physiques sur l'envergure
    w_map, a_map = _modal_to_physical_fields(par, qw_t, qa_t)  # renvoie w(y,t) alpha(y,t)

    # puis on prend les valeurs temporelles en bout d'aile [tip_idx]
    # Extraction au bout (y=s) = dernier point de par.y
    tip_idx = -1
    w_tip = w_map[:, tip_idx] if w_map.size else np.zeros(nt)
    alpha_tip = a_map[:, tip_idx] if a_map.size else np.zeros(nt)

    # Prétraitement pour FFT
    xw = w_tip.copy()
    xa = alpha_tip.copy()
    if detrend:
        xw = xw - np.mean(xw)
        xa = xa - np.mean(xa)
    if window:
        win = np.hanning(nt)
        xw = xw * win
        xa = xa * win

    # Zero-padding
    n_fft = int(2 ** np.ceil(np.log2(nt))) * int(max(1, zero_pad))

    # FFT mono-côté et axe fréquentiel (Hz)
    W = np.fft.rfft(xw, n=n_fft)
    A = np.fft.rfft(xa, n=n_fft)
    f = np.fft.rfftfreq(n_fft, d=dt)

    # Amplitude (échelle simple; pour amplitude single-sided, on multiplie par 2/N)
    scale = 2.0 / nt
    W_mag = scale * np.abs(W)
    A_mag = scale * np.abs(A)

    # Limites de fréquence
    if freq_max is None:
        freq_max = f[-1]
    f_sel = f <= freq_max

    # Figure 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    if U>=0: # si U = None alors ça renvoie false
        fig.suptitle(f'U = {U} m/s')

    # w(y=s, t)
    axes[0, 0].plot(t, w_tip, color='tab:blue')
    axes[0, 0].set_title("w(y=s, t)")
    axes[0, 0].set_xlabel("t [s]")
    axes[0, 0].set_ylabel("w_tip")
    axes[0, 0].grid(True)

    # FFT w
    axes[0, 1].plot(f[f_sel], W_mag[f_sel], color='tab:blue')
    axes[0, 1].set_title("FFT(w(y=s, t))")
    axes[0, 1].set_xlabel("f [Hz]")
    axes[0, 1].set_ylabel("|W(f)|")
    axes[0, 1].set_xlim(0, 50)
    axes[0, 1].grid(True)

    # alpha(y=s, t)
    axes[1, 0].plot(t, alpha_tip, color='tab:orange')
    axes[1, 0].set_title("alpha(y=s, t)")
    axes[1, 0].set_xlabel("t [s]")
    axes[1, 0].set_ylabel("alpha_tip [rad]")
    axes[1, 0].grid(True)

    # FFT alpha
    axes[1, 1].plot(f[f_sel], A_mag[f_sel], color='tab:orange')
    axes[1, 1].set_title("FFT(alpha(y=s, t))")
    axes[1, 1].set_xlabel("f [Hz]")
    axes[1, 1].set_ylabel("|A(f)|")
    axes[1, 1].set_xlim(0, 50)
    axes[1, 1].grid(True)

    plt.show()

    if return_data:
        return t, w_tip, alpha_tip, f[f_sel], W_mag[f_sel], f[f_sel], A_mag[f_sel]
