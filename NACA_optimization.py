#%%
import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultMultiObjectiveTermination

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
n_ieq_constr = 1 # nombre de contraintes à respecter (G)

problem_optim_NACA  = ProblemOptim(n_var = para_interval.shape[0], n_obj = n_obj, 
                                    n_ieq_constr = n_ieq_constr, # nombre de contrainted par inéquation
                                    xl=xl, xu=xu, # bornes inf et sup de tous nos paramètres variables
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
res = minimize(problem_optim_NACA,   # herite forcement de la classe Problem et doit présenter une fonction _evaluate() bien définie
                algorithm_NSGAII, #objet algo optim, il définit le type d'algo d'optim utilisé
                ("n_gen", 5), # terminaison : critère d'arrêt pour l'algo, ici on fait n génération et on s'arrête
                verbose=True, # pour afficher ou non les info du processus d'optim, utile pour debug
                seed=1) # permet de reproduire les résultats en fixant la séquence aléatoire
                        # save_history(optionnel) sur True l'historique de générations sera enregistré pour un post traitement

# %%
