#%%
import numpy as np
import NACA
import ROM
import plotter
from input import ModelParameters
from data_manager import save_modal_data

from _functions_AS_optim import map_to_physical

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultMultiObjectiveTermination, DefaultSingleObjectiveTermination

from pymoo.optimize import minimize
from ProblemOptim import ProblemOptim


coeff_low, coeff_high = 0.6, 1.4
para_interval = np.array([
    [0.0, 1.0],                             # u x_ea/c [0.15,0.5]
    [0.0, 1.0],                             # v facteur d'écart du CG à 0.8c
    [coeff_low * 366, coeff_high * 366],
    [coeff_low * 78, coeff_high * 78]
])

xl = para_interval[:,0]
xu = para_interval[:,1]
n_obj = 1 # nombre d'objectifs à optim en même temps (F)
n_ieq_constr = 0 # nombre de contraintes à respecter (G)

problem_optim_NACA  = ProblemOptim(n_var = para_interval.shape[0], n_obj = n_obj, 
                                    n_ieq_constr = n_ieq_constr, # nombre de contrainted par inéquation
                                    xl=xl, xu=xu, # bornes inf et sup de tous nos paramètres variables
)

algorithm_GA = GA(
        pop_size=20,
        sampling=LHS(),
        eliminate_duplicates=True,
        crossover=SBX(prob=0.9, eta=20),
        mutation=PolynomialMutation(prob=1/para_interval.shape[0], eta=20)
        ) 

algorithm_NSGAII=NSGA2( # NSGA hérite de GeneticALgorithm qui hérite de Algorithm
                pop_size=20, #taille de l'échantillon initial
                n_offsprings=10, #taille de chaque échantillon d'enfants créés à partir de la pop actuelle par le biais des opérateurs de croisement et mutation
                                    # un n_offsprings + élevé permet une exploration plus large de l'espace mais prend + de temps
                                    # dans GeneticAlgorithm
                                    # if self.n_offsprings is None:
                                    #      self.n_offsprings = pop_size
                sampling=FloatRandomSampling(), #Méthode d'échantillonage pour la population initiale : échantillonage aléatoire choisi, c'est comme ça par défaut si on renseigne rien 
                crossover=SBX(prob=0.9,eta=20), #Lois de probabilité définissant les opérateurs de croisement (par défait eta=15, prob = 0.9)
                mutation=PolynomialMutation(eta=20), #Lois de probabilité définissant les opérateurs de mutation
                eliminate_duplicates=True, # ce n'est pas un attribut de NSGA2 mais bien de la class dont il hérite GeneticAlgorithm, probablement gérer par le **kwargs
                termination=DefaultMultiObjectiveTermination( # une seule des conditions suivantes entraîne l'arrêt de l'algo
                                xtol=0.0005, # si les changements des paramètres entres gen sont inf à xtol alors on s'arrête. VALEUR de variation MOYENNE ????? ou par PARA ???
                                cvtol=1e-8, # par rapport aux contraintes ?
                                ftol=0.005, # si les changements dans les valeurs des obj entre 2 gen est < a ftol alors on s'arrête
                                n_max_gen=10) # nbr max de gen
                )

algorithm_DE = DE(
    pop_size=20,
    sampling=LHS(),#Méthode d'échantillonage pour la population initiale
    #LHS Latin Hypercub Sampling, couvre uniformement l'espace de var (souvent mieux qu'un echantillon aléatoire ou ça peut être regroupé au même endroit par hasard)
    variant="DE/rand/1/bin", # stratégie de mutation utilisée DE/<base>/<num_diff>/<crossover>
    # rand : vecteur de base x_r est choisi random dans pop
    # 1 : une seule diff (x_a-x_b) utilisée pour construire le mutant v=x_r + F(x_a-x_b)
    # bin : crossover binaire (chq composant du mutant est kept w/ prop CR)
    # autre possibilité : "DE/best/1/bin" : plus exploitant (le meilleur individu comme base → plus rapide, mais risque local minima).
    CR=0.7, #crossover rate: lors du crossover bin chq var du mutant vi remplace celle du parent xi avec prob=CR
    # fonction est très corrélée entre variables, un CR élevé aide
    dither="vector", # modulation du facteur d'amp F
    # dither = None : F constant 0.5
    # dither = 'scalar' : un même F aleatoire est tiré pour toute la pop à chaque génération
    # dither = 'vector : un F diff est tiré pour chq individu -> plus de diversité (meilleur choix pour éviter une convergence prématurée)
    jitter=False # pour perturber F, sur False si dither déjà activé
)

'''CMA-ES tire des individus depuis une loi normale multivariée N(m, sigma²C) dont sigma est la taille de pas globale et C la matrice de covariance'''
algorithm_CMAES = CMAES(
    x0 = np.random.random(problem_optim_NACA.n_var),  # point initial
    sigma = 0.3,  # écart-type initial (taille des pas)
    population_size = None,  # si None, valeur par défaut selon n_var
    sampling = LHS(),  # initialisation de la population
    restarts = 0,  # nombre de redémarrages (utile pour éviter minima locaux)
    restart_from_best = True,  # conserve le meilleur individu au redémarrage
    # tolfun = 1e-6,  # tolérance sur la variation de la fonction objectif
    # tolfunhist = 1e-12,  # tolérance sur l’historique des valeurs f(x)
    # tolx = 1e-12,  # tolérance sur la variation des positions
) #sigma et restarts sont spécifiques à CMAES





#%%______MINIZATION______________________________________________________________________________

'''
termination = DefaultSingleObjectiveTermination(
    xtol=1e-6,      # tolérance sur les variables
    ftol=1e-6,      # tolérance sur la valeur du meilleur F de la génération, d'une gen à la suivant si ça a pas changé de plus de tol sur n_last gen alors ça stop
    n_last=20,      # nb de générations consécutives à vérifier
    n_max_gen=20,  # borne max
    # n_max_evals=1e5
)
'''

algorithms = {
    "GA": algorithm_GA,
    "DE": algorithm_DE,
    "CMAES": algorithm_CMAES
}

algorithm_name = "GA"  
algorithm = algorithms[algorithm_name]


res = minimize(
                problem_optim_NACA, # herite forcement de la classe Problem et doit présenter une fonction _evaluate() bien définie
                algorithm,       #objet algo optim, il définit le type d'algo d'optim utilisé
                ('n_gen', 20),       # terminaison : critère d'arrêt pour l'algo, ici on fait n génération et on s'arrête
                verbose=True,       # pour afficher ou non les info du processus d'optim, utile pour debug
                seed=2              # permet de reproduire les résultats en fixant la séquence aléatoire
                )                   # save_history(optionnel) sur True l'historique de générations sera enregistré pour un post traitement
'''
res.X : X*
res.F : objective value associated to X F(X)
res.G : same for constraints G(X)
res.CV : violation des contraintes (0 normalement)
res.exec_time : temps de calcul 
'''

np.savez('data/res_'+algorithm_name,resX=res.X, resF = res.F)





#%%_____test optimal solution_______________________________________________________________________
algorithm_name = "DE"
data = np.load('data/res_'+algorithm_name+'.npz')
X_opt = map_to_physical(data['resX'])
s, c = 2.0, 0.2
m = 2.4
eta_w = 0.005
eta_alpha = 0.005
XX = [X_opt[0]*c,X_opt[1]*c,X_opt[2],X_opt[3]]
model = ModelParameters(s, c, x_ea=X_opt[0]*c, x_cg=X_opt[1]*c, m=m, EIx=X_opt[2], GJ=X_opt[3], eta_w=eta_w, eta_alpha=eta_alpha,model_aero= 'Theodorsen')
model.Umax=22
model.steps=80
f, damping, *_ = ROM.ModalParamDyn(model)
save_modal_data(f = f, damping = damping, model_params=model,out_dir='data', filename=f'model_optim_{algorithm_name}.npz')
plotter.plot_modal_data_single(npz_path=f'data/model_optim_{algorithm_name}.npz' )

model.airfoil.plot_naca00xx_section()
# %%
data = np.load('data/res_'+'DE'+'.npz')
X_opt_DE = map_to_physical(data['resX'])
X_opt_DE[0] *= c
X_opt_DE[1] *= c

data = np.load('data/res_'+'GA'+'.npz')
X_opt_GA = map_to_physical(data['resX'])
X_opt_GA[0] *= c
X_opt_GA[1] *= c

# %%
plotter.plot_modal_data_two('data/model_optim_GA.npz','data/model_optim_DE.npz')
# %%
