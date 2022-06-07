import random
from code import Params

from deap import base, creator, tools


class Config():

    def __init__(self, evaluator):
        creator.create("Fitness", base.Fitness, weights=evaluator.problem_type)
        creator.create("Individual", list, fitness=creator.Fitness)

        self.toolbox = base.Toolbox()
        self.toolbox.register("indices", random.randint,
                              0, Params.N_ALGORITHMS)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.indices, n=Params.INDIVIDUAL_SIZE)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        self.toolbox.register("evaluate", evaluator.evaluate)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.50)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.25)
        # self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("select", tools.selNSGA3, ref_points=tools.uniform_reference_points(
            nobj=6, p=12))

        self.pop = self.toolbox.population(n=Params.POPULATION_SIZE)
