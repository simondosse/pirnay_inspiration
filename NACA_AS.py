#%%
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
from input import ModelParameters
import ROM
import plotter
import NACA
from morris_method import generate_trajectories,resultat, funnel_graph
from SALib.sample import saltelli
from SALib.analyze import sobol

'''
This script is used to perform sensitivity analysis (Method or Sobol)
for NACA section's parameters
'''




#%% ------- MORRIS ----------------------------
c_low, c_high = 0.6, 1.4

para_interval = np.array([
    [0.15, 0.4],
    [0.4, 0.8],
    [c_low * 366, c_high * 366],
    [c_low * 78, c_high * 78]
])
nb_para = para_interval.shape[0]

problem= {
    'num_vars': nb_para,  # Nombre de paramètres
    'names': [r'x_{ea}/c',r'x_{cg}/c',r'EI_z',r'GJ'],  # Noms des paramètres en LaTeX
    'bounds': para_interval # Plages
}

# Sampling
X = generate_trajectories(problem = problem, nb_traj_opti=3) # watch out the seed setup (seed= 1, we fixe the randomness)

# F(X)
s, c = 2.0, 0.2
m = 2.4
eta_w = 0.005
eta_alpha = 0.005

F = np.zeros((len(X),2))
damping_box = np.zeros((len(X),100,2))
f_box = np.zeros((len(X),100,2))

for i in range(len(X)):

    x_ea = X[i][0]*c # *c because we are dealing with adimensionnal parameter
    x_cg = X[i][1]*c
    res = NACA.inertia_mass_naca0015(c=c, mu=m, N=4000, span=s, xcg_known=x_cg)
    I_alpha=res.Jz_mass+m*abs(x_cg-x_ea)**2 # parallel axis theorem to get torsional inertia about elastic axis
    EIx = X[i][2]
    GJ = X[i][3]

    model = ModelParameters(s, c, x_ea, x_cg, m, I_alpha, EIx, GJ, eta_w, eta_alpha, None, None, None, 'Theodorsen')
    f, damping, _ = ROM.ModalParamDyn(model)
    damping_box[i,:,0] = damping[:,0]
    damping_box[i,:,1] = damping[:,1]
    f_box[i,:,0] = f[:,0]
    f_box[i,:,1] = f[:,1]

    Uc,slope_damping = ROM.damping_crossing_slope(U = model.U, damping = damping[:,0])

    F[i][0]=Uc
    F[i][1]=slope_damping
    print(f'F[{i+1}] / F[{len(X)}]')

np.savez('data/F_morris',F=F)

fig1, ax1 = plt.subplots()
for i in range(len(X)):
    if i == 0:
        ax1.plot(model.U,damping_box[i,:,0],color = 'blue',label='X(0) - d[0]')
        ax1.plot(model.U,damping_box[i,:,1],color = 'red',label='X(0) - d[1]')
    else:
        ax1.plot(model.U,damping_box[i,:,0],color = 'blue')
        ax1.plot(model.U,damping_box[i,:,1],color = 'red')
ax1.legend()
ax1.grid(True, linewidth=0.3, alpha=0.5)
ax1.set_title('damping(U)')
plt.show()



fig, ax = plt.subplots()
for i in range(len(X)):
    if i == 0:
        ax.plot(model.U,f_box[i,:,0],color = 'blue',label='X(0) - d[0]')
        ax.plot(model.U,f_box[i,:,1],color = 'red',label='X(0) - d[1]')
    else:
        ax.plot(model.U,f_box[i,:,0],color = 'blue')
        ax.plot(model.U,f_box[i,:,1],color = 'red')
ax.legend()
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.set_title('f(U)')
plt.show()



#%% Results
F = np.load('data/F_morris.npz')['F']
obj_names = np.array([r'$U_c$',r'$\frac{d\zeta}{dU}$'])
'''
- We process one objective at a time
- sigma and mu* for each objective
- bar diagram
'''
# On traite nos objectifs un par un
# On trace les sigma et mu* pour chaque objectif
# On trace les diagrammes en barres pour les mu* pour chaque paramètre pour chaque objectif
for i in range(F.shape[1]):
    results, elem = resultat(problem,
                             X, 
                             F[:,i],
                             nb_lvl=4,
                             title = obj_names[i],
                             cas = 's fixed')

#%% ------- SOBOL ----------------------------

def map_to_physical(X_uv):
    '''
    Function go to back to physical parameters as we changed variables
    
    Parameters
    ----------
    X_uv : array
        list N*nb_para, samples to be evaluated in the UV space
    
    Returns
    ---------
        np.array
        samples that have be evaluated F(X)
    
    
    '''
    x_ea_low, x_ea_high = 0.15 , 0.5 # x_ea can't go beyond the half-chord
    x_cg_high = 0.8
    # idk if we should add a lower boundary for x_cg (sometimes x_cg_low shoudn't be x_ea)

    u_ea, v_cg, EI, GJ = X
    x_ea = x_ea_low + (x_ea_high - x_ea_low) * u_ea
    xcg_min = x_ea
    x_cg = xcg_min + (x_cg_high - xcg_min) * v_cg

    X = np.array([x_ea, x_cg, EI, GJ])
    return 

c_low, c_high = 0.3, 3

para_interval = np.array([
    [0.0, 1.0],
    [0.0, 1.0],
    [c_low * 366, c_high * 366],
    [c_low * 78, c_high * 78]
])
nb_para = para_interval.shape[0]

problem_uv = {
    'num_vars': nb_para,  # Nombre de paramètres
    'names': [r'x_{ea}/c',r'x_{cg}/c',r'EI_z',r'GJ'],  # Noms des paramètres en LaTeX
    'bounds': para_interval # Plages
}

N = 1024  # base sample size
X_uv = saltelli.sample(problem, N, calc_second_order=False) # calc_second_order=False divides /2 the number of evaluations
X = map_to_physical(X_uv=X_uv)



# %%
