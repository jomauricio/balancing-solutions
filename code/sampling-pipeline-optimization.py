import random
import numpy
import warnings
import csv

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import cohen_kappa_score
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import OneSidedSelection
from imblearn.combine import SMOTETomek


# Carregar Base inglês ou portugues
                                # RandomForest Parameters
# baseDados = "CoI-en"          # n_estimators=1000, max_features=10
# baseDados = "CoI-pt"          # n_estimators=500, max_features=6

# database = pd.read_csv(baseDados +".csv")
# target = database['classe'].tolist()
# database = database.drop(['classe'], 1)
# database = database.drop(['ID'], 1)
# data = database.as_matrix()

baseDados = "Bases_Individuais"
# test = pd.read_csv("en - en.csv")
# train = pd.read_csv("pt - pt.csv")
#
# test_target = test['classe'].tolist()
# test = test.drop(['classe'], 1)
# test_data = test.as_matrix()
#
# train_target = train['classe'].tolist()
# train = train.drop(['classe'], 1)
# train_data = train.as_matrix()

# Carregar bases treino e teste individuais

# baseDados = "Bases_Individuais"
# test = pd.read_csv("test_data.csv")
# train = pd.read_csv("train_data.csv")
#
# test_target = test['category'].tolist()
# test = test.drop(['id'], 1)
# test = test.drop(['category'], 1)
# test_data = test.as_matrix()
#
# train_target = train['category'].tolist()
# train = train.drop(['id'], 1)
# train = train.drop(['category'], 1)
# train_data = train.as_matrix()


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

SIZE = 5

toolbox = base.Toolbox()
toolbox.register("indices", random.randint, 0, 6)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.indices, n=SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'pop', 'hof'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), hof=halloffame, **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), pop=population, hof=halloffame, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def my_evaluate(individual):
    #test = pd.read_csv("test_data.csv")
    #train = pd.read_csv("train_data.csv")
    test = pd.read_csv("en - en.csv")
    train = pd.read_csv("pt - pt.csv")

    # test_target = test['classe'].tolist()
    # test = test.drop(['classe'], 1)
    # test_data = test.as_matrix()
    test_data, test_target = test.iloc[:,0:53], test["classe"]

    # train_target = train['classe'].tolist()
    # train = train.drop(['classe'], 1)
    # train_data = train.as_matrix()
    train_data, train_target = train.iloc[:,0:53], train["classe"]

    output_train_data = train_data
    output_train_target = train_target
    v=0
    try:
        for feature in individual:
            if (feature == 0):
                v+=1
            elif (feature == 1):
                output_train_data, output_train_target = NearMiss().fit_resample(output_train_data, output_train_target)
            elif (feature == 2):
                output_train_data, output_train_target = EditedNearestNeighbours().fit_resample(output_train_data, output_train_target)
            elif (feature == 3):
                output_train_data, output_train_target = SMOTE().fit_resample(output_train_data, output_train_target)
            elif (feature == 4):
                output_train_data, output_train_target = TomekLinks().fit_resample(output_train_data, output_train_target)
            elif (feature == 5):
                output_train_data, output_train_target = OneSidedSelection().fit_resample(output_train_data, output_train_target)
            elif (feature == 6):
                output_train_data, output_train_target = SMOTETomek().fit_resample(output_train_data, output_train_target)
        if v < 4:
            # pipeline_a = make_pipeline(t[0], t[1], t[2], t[3], t[4], RandomForestClassifier(n_estimators=100, max_features=6))
            clf = RandomForestClassifier(n_estimators=200, max_features=6, warm_start=True, oob_score=True)
            clf.fit(output_train_data, output_train_target)
            predict = clf.predict(test_data)
            acc = accuracy_score(test_target, predict)
            # fscore = f1_score(y_test, predict, average="weighted")
            # gm = geometric_mean_score(y_test, predict, average="weighted")
            # score = pipeline_a.decision_function(X_test)
            # auroc = roc_auc_score(y_test, score, average="weighted")
            # kappa = cohen_kappa_score(test_target, predict)
        else:
            acc = 0
            # kappa = 0
    except:
        acc = 0
        # kappa = 0
        # fscore = 0
        # gm = 0
        # auroc = 0
    #return acc,# fscore, gm #, auroc,
    return acc,


toolbox.register("evaluate", my_evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.25)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed()

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis = 0)
    stats.register("std", numpy.std, axis = 0)
    stats.register("min", numpy.min, axis = 0)
    stats.register("max", numpy.max, axis = 0)

    # pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=5, stats=stats, halloffame=hof, verbose=True)
    pop, log = eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=5, stats=stats, halloffame=hof, verbose=True)
    return pop, log, hof


if __name__ == "__main__":
    f = open("RandomForest_GA_30por1000_ACC" + "_Base-" + baseDados + ".txt", "w")
    warnings.filterwarnings("ignore")


    test = pd.read_csv("en - en.csv")
    train = pd.read_csv("pt - pt.csv")
    test_data, test_target = test.iloc[:, 0:53], test["classe"]
    train_data, train_target = train.iloc[:, 0:53], train["classe"]

    clf = RandomForestClassifier(n_estimators=200, max_features=6, warm_start=True, oob_score=True)
    clf.fit(train_data, train_target)
    predict = clf.predict(test_data)
    print(accuracy_score(test_target, predict))


    results = main()
    df_log = pd.DataFrame(results[1])
    df_log.to_csv("LogBook_RandomForest_GA_30por1000_ACC" + "_Base-" + baseDados + ".csv", index=False)
    f.write(str(results[2]))
    f.close()


