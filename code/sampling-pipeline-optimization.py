import random
import numpy
import warnings
from deap import tools
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

from algorithm.BasicGA import BasicGA
from algorithm.Config import Config
from evaluation_function.Fitness import Fitness

def main():
    random.seed()

    evaluator = Fitness()
    config = Config(evaluator)

    algorithm = BasicGA()

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis = 0)
    stats.register("std", numpy.std, axis = 0)
    stats.register("min", numpy.min, axis = 0)
    stats.register("max", numpy.max, axis = 0)

    # pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=5, stats=stats, halloffame=hof, verbose=True)
    pop, log = algorithm.run(config, cxpb=0.5, mutpb=0.3, ngen=5, stats=stats, halloffame=hof, verbose=True)
    return pop, log, hof


if __name__ == "__main__":
    f = open("RandomForest_GA_30por1000_ACC" + "_Base.txt", "w")
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
    df_log.to_csv("LogBook_RandomForest_GA_30por1000_ACC" + "_Base.csv", index=False)
    f.write(str(results[2]))
    f.close()


