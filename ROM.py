import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import kv

''' General functions to compute the mode shapes and modal matrices '''

def bendingModeShapes(par):
    """
    Build the bending mode shapes of the wing.
    """
    n = par.Nw

    if n < 1:   # Simply check if n is a positive integer
        raise ValueError("n must be a positive integer")

    if par.has_tip is True:     # Case with added mass on the tip, beta values are computed by evaluating the characteristic equation 

        b = np.linspace(0, n*np.pi, n*10000)
        func = 1 + (np.cos(b * par.s) * np.cosh(b * par.s)) + (par.Mt * b/par.m) * (np.sinh(b * par.s)*np.cos(b * par.s) - np.sin(b * par.s)*np.cosh(b * par.s))

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
            if func[i] * func[i+1] < 0:
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

        modal_mass =  np.trapezoid(par.m*phi**2, par.y)

        phi_normalized.append(phi)#/np.sqrt(modal_mass))
        phi_dot_normalized.append(phi_dot)#/np.sqrt(modal_mass))
        phi_dotdot_normalized.append(phi_dotdot)#/np.sqrt(modal_mass))
        
    return phi_normalized, phi_dot_normalized, phi_dotdot_normalized
    
def torsionModeShapes(par):
    """
    Build the torsion mode shapes of the wing.
    """
    n = par.Nalpha

    if n < 1:       # Simply check if n is a positive integer
        raise ValueError("n must be a positive integer")

    if par.has_tip is True:         # Case with added mass on the tip, beta values are computed by evaluating the characteristic equation
        b = np.linspace(0.01, n*np.pi, n*10000)
        func = np.tan(b * par.s) - (par.I_alpha * par.s / par.I_alpha_t) * 1/(b * par.s)
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
        phi_normalized.append(phi)

        phi_dot = np.cos(beta[i] * par.y) * beta[i]
        phi_dot_normalized.append(phi_dot)

        phi_dotdot = - np.sin(beta[i] * par.y) * (beta[i] ** 2)
        phi_dotdot_normalized.append(phi_dotdot)

    return phi_normalized, phi_dot_normalized, phi_dotdot_normalized

def modalMatrices(par):
    """
    Build the modal matrices to avoid redundant computations.
    """

    phi_w, _, _ = bendingModeShapes(par)
    phi_alpha, _, _ = torsionModeShapes(par)

    phi_w_interp = [interp1d(par.y, phi_w[i], kind='cubic', fill_value="extrapolate") for i in range(par.Nw)]
    phi_alpha_interp = [interp1d(par.y, phi_alpha[i], kind='cubic', fill_value="extrapolate") for i in range(par.Nalpha)]

    ''' Initialize modal matrices '''
    phi_ww = np.zeros((par.Nw, par.Nw))
    phi_alphaalpha = np.zeros((par.Nalpha, par.Nalpha))
    phi_walpha = np.zeros((par.Nw, par.Nalpha))

    for i in range(par.Nw):
        for j in range(par.Nw):
            def integrand(y):
                return phi_w_interp[i](y) * phi_w_interp[j](y)

            phi_ww[i, j] = quad(integrand, 0, par.s)[0]

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
    Function used to generalize the computation of the mass and stiffness matrices (structural). It can only be used for matrices made of element of the 
    form X = int(arg * phi1 * phi2) and Xtip = argtip * phi1[s] * phi2[s] if the tip mass is considered.
    """

    X = np.zeros((len1,len2))
    Xtip = np.zeros((len1,len2)) 

    phi1_interp = [interp1d(par.y, phi1[i], kind='cubic', fill_value="extrapolate") for i in range(len1)]
    phi2_interp = [interp1d(par.y, phi2[j], kind='cubic', fill_value="extrapolate") for j in range(len2)]

    for i in range(len1):
        for j in range(len2):
            def integrand(y):
                return arg * phi1_interp[i](y) * phi2_interp[j](y)

            X[i, j] = quad(integrand, 0, par.s)[0]

            if par.has_tip is True:
                Xtip[i, j] = argtip * phi1[i][-1] * phi2[j][-1]

    return X + Xtip

''' Computation of the structural matrices '''

def getStructuralMassMatrix(par):
    """
    Compute the structural mass matrix of the wing.
    """

    phi_w, _, _ = bendingModeShapes(par)
    phi_alpha, _, _ = torsionModeShapes(par)

    Mww = getMatrix(phi_w , phi_w , par.Nw , par.Nw , par.m , par.Mt , par)
    Mwalpha = getMatrix(phi_w , phi_alpha , par.Nw , par.Nalpha , par.m*par.x_alpha , par.Mt*par.x_t , par)
    Malphaalpha = getMatrix(phi_alpha , phi_alpha , par.Nalpha , par.Nalpha , par.I_alpha , par.I_alpha_t , par)

    M = np.block([[Mww, Mwalpha], [ Mwalpha.T, Malphaalpha]])

    return M
    
def getStructuralStiffness(par):
    """
    Compute the structural stiffness matrix of the wing.
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
    """

    phi_ww, phi_alphaalpha, phi_walpha = modalMatrices(par)
    Ka = 0.5 * par.rho_air * (U**2) * par.c * np.block([[0 * phi_ww , par.dCn * phi_walpha], [0 * phi_walpha.T , - par.c * par.dCm * phi_alphaalpha]])
  
    return Ka 

def TheodorsenFunction(k):
    """
    Compute the Theodorsen function for a given reduced frequency.
    """

    # if k <= 1/2 :
    #     C =  1 - ((0.165)/(1 - 0.0455j/k)) - ((0.335)/(1 - 0.3j/k))

    # else :
    #     C =  1 - ((0.165)/(1 - 0.041j/k)) - ((0.335)/(1 - 0.32j/k))

    # return np.real(C), np.imag(C)

    K0 = kv(0, 1j * k)
    K1 = kv(1, 1j * k)

    C = K1 / (K0 + K1)

    return np.real(C), np.imag(C)

def CooperWrightNotation(par,k):
    """
    Convert the Theodorsen function into the Cooper-Wright notation.
    """

    F,G = TheodorsenFunction(k)

    L_w_dot_dot = np.pi
    L_w_dot = par.dCn * F
    L_w = - par.dCn * k * G

    L_alpha_dot_dot = - np.pi * par.a
    L_alpha_dot = par.dCn * ( F * (1/2 - par.a) + (G/k)) + np.pi
    L_alpha = par.dCn * (F - k * G * (1/2 - par.a) )

    M_w_dot_dot = np.pi * par.a
    M_w_dot = 2 * par.dCm * F
    M_w = - 2 * par.dCm * k * G 

    M_alpha_dot_dot = - np.pi * (1/8 + (par.a**2))
    M_alpha_dot = 2 * par.dCm * (F * (1/2 - par.a) + (G/k)) + (-np.pi * (1/2 - par.a))
    M_alpha =  2 * par.dCm * (F - k * G * (1/2 - par.a))

    return L_w_dot_dot, L_w_dot, L_w, L_alpha_dot_dot, L_alpha_dot, L_alpha, M_w_dot_dot, M_w_dot, M_w, M_alpha_dot_dot, M_alpha_dot, M_alpha

def TheodoresenAeroModel(par,U,omega):
    """
    Build the added aerodynamic mass, stiffness and damping matrices predicted by the Theodoresen model.
    """

    k = par.b * omega / U
    L_w_dot_dot, L_w_dot, L_w, L_alpha_dot_dot, L_alpha_dot, L_alpha, M_w_dot_dot, M_w_dot, M_w, M_alpha_dot_dot, M_alpha_dot, M_alpha = CooperWrightNotation(par,k)
    phi_ww, phi_alphaalpha, phi_walpha = modalMatrices(par)

    Ka = par.rho_air * (U**2) * np.block([[L_w * phi_ww , L_alpha * par.b * phi_walpha], [- M_w * par.b * phi_walpha.T, - M_alpha * (par.b**2) * phi_alphaalpha]])
    Ca = par.rho_air * U * par.b *np.block([[L_w_dot * phi_ww , L_alpha_dot * par.b * phi_walpha], [- M_w_dot * par.b * phi_walpha.T, - M_alpha_dot * (par.b**2) * phi_alphaalpha]])
    Ma = par.rho_air * (par.b**2) * np.block([[L_w_dot_dot * phi_ww , L_alpha_dot_dot * par.b * phi_walpha], [- M_w_dot_dot * par.b * phi_walpha.T, - M_alpha_dot_dot * (par.b**2) * phi_alphaalpha]])

    return Ka, Ca, Ma

''' Eigenvalue problem and modal parameters extraction '''

''' At rest'''

def ModalParamAtRest(par):

    M = getStructuralMassMatrix(par)
    K = getStructuralStiffness(par)
    C = getStructuralDamping(par)

    Minv = np.linalg.inv(M)

    A = np.block([[-Minv @ C, -Minv @ K], [np.eye(par.Nw + par.Nalpha), np.zeros((par.Nw + par.Nalpha,par.Nw + par.Nalpha))]])
    eigVal, _ = np.linalg.eig(A)

    aux = np.imag(eigVal)
    idx = np.argsort(abs(aux))
    eig = aux[idx]

    return eig[::2] / (2*np.pi)

''' Dynamic '''

def stateMatrixAero(par,U,omega):
    """
    Build the state matrix of the system (structural and aerodynamic consideration).
    """

    M = getStructuralMassMatrix(par)
    C = getStructuralDamping(par)
    K = getStructuralStiffness(par)

    if par.model_aero == 'QuasiSteady':
        Ka = QuasiSteadyAeroModel(par,U)
        K = K + Ka

    elif par.model_aero == 'Theodorsen':
        Ka , Ca , Ma = TheodoresenAeroModel(par,U,omega)
        K = K + Ka
        C = C + Ca
  
    else:
        raise ValueError("Model not recognized")
    
    Minv = np.linalg.inv(M)
    A = np.block([[-Minv @ C, -Minv @ K], [np.eye(par.Nw + par.Nalpha), np.zeros((par.Nw + par.Nalpha,par.Nw + par.Nalpha))]])
    
    return A

def ModalParamDyn(par):
    ''' 
    Compute the modal parameters of the system for a range of wind speeds.
    '''

    U = par.U
    omega = ModalParamAtRest(par) * 2 * np.pi
 
    f = np.zeros((len(U),2))
    damping = np.zeros((len(U),2))
    realpart = np.zeros((len(U),2))

    previous_omega = (omega[1] + omega[2])/2
    prev =[]

    for i in range(len(U)):
        prev.append(previous_omega)
        
        A = stateMatrixAero(par,U[i],previous_omega)
        eigVal, _ = np.linalg.eig(A)
        eigVal = eigVal[::2]

        eig = np.imag(eigVal)
        idx = np.argsort(abs(eig))
        eig = eig[idx]

        p = np.real(eigVal)[idx]
        eta = - np.sign(p) * np.sqrt(1 / (1 + (eig/p)**2))

        previous_omega = (eig[1] + eig[2])/2

        f[i,0] = eig[1] / (2*np.pi)
        f[i,1] = eig[2] / (2*np.pi)

        damping[i,0] = eta[1]
        damping[i,1] = eta[2]

        realpart[i,0] = p[1]
        realpart[i,1] = p[2]

    return f, damping , realpart
