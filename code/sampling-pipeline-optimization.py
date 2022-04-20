import random
import warnings

import numpy
from algorithm.BasicGA import BasicGA
from algorithm.Config import Config
from deap import tools
from evaluation_function.Fitness import Fitness

from . import Params


def write_results(results):
    with open("code/results.txt", "w") as external_file:
        print(results, file=external_file)
        external_file.close()


def main():
    random.seed()

    evaluator = Fitness()
    config = Config(evaluator)

    algorithm = BasicGA()

    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    pop, log = algorithm.run(config, cxpb=0.5, mutpb=0.5, ngen=Params.NGENERATIONS, stats=stats, halloffame=hof,
                             verbose=True)

    return pop, log, hof


if __name__ == "__main__":
    # f = open("RandomForest_GA_"+ str(Params.POPULATION_SIZE) + "por" + str(Params.NGENERATIONS) +"_ACC.txt", "w")
    warnings.filterwarnings("ignore")
    resulsts = main()
    write_results(resulsts)
    print(resulsts)

    # Ler sequencias do dicionario(.csv) e montar grafico com pca
    #
    # sequence = pd.read_csv("sequences3.csv", names=['keys', 'values'])
    # keys = []
    # for key in sequence['keys']:
    #     chave = (list(filter(lambda x: x not in ['[', ']', chr(39), ',', ' '], key)))
    #     chave = [int(x) for x in chave]
    #     for i in range(5 - len(chave)):
    #         chave.insert(0, 0)
    #     keys.append(chave)
    # df = pd.DataFrame(keys, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    # data = df.iloc[:, 0:5]
    #
    # pca = PCA(n_components=1)
    # principalComponents = pca.fit_transform(data)
    # data = zip(keys, principalComponents, sequence['values'])
    # data = sorted(data, key=lambda dat: dat[1])
    # chaves = []
    # values = []
    # for dat in data:
    #     chaves.append(str(dat[0]))
    #     values.append(dat[2])
    # plt.plot(chaves, values)
    # plt.show()
    #
    #
    # Coleta da base resultante
    #
    # test = Params.BASE_TEST
    # train = Params.BASE_TRAIN
    #
    #
    # test_data, test_target = test.iloc[:, 1:52], test["classe"]
    # train_data, train_target = train.iloc[:, 1:52], train["classe"]
    #
    # ind = [3]
    # # output_train_data = Normalizer().fit_transform(train_data)
    # output_train_data = train_data
    # output_train_target = train_target
    # for feature in ind:
    #     if (feature == 1):
    #         output_train_data, output_train_target = NearMiss().fit_resample(output_train_data,
    #                                                                          output_train_target)
    #     elif (feature == 2):
    #         output_train_data, output_train_target = EditedNearestNeighbours().fit_resample(output_train_data,
    #                                                                                         output_train_target)
    #     elif (feature == 3):
    #         output_train_data, output_train_target = SMOTE().fit_resample(output_train_data,
    #                                                                       output_train_target)
    #     elif (feature == 4):
    #         output_train_data, output_train_target = TomekLinks().fit_resample(output_train_data,
    #                                                                            output_train_target)
    #     elif (feature == 5):
    #         output_train_data, output_train_target = OneSidedSelection().fit_resample(output_train_data,
    #                                                                                   output_train_target)
    # # print(output_train_data)
    # df = pd.DataFrame(data=output_train_data, columns=train_data.columns)
    # df['classe'] = output_train_target
    # df.to_csv("SMOTE_train-en_old-Normalized2.csv")
    # clf = RandomForestClassifier(n_estimators=1000, max_features=6, warm_start=True, oob_score=True)
    # clf.fit(output_train_data, output_train_target)
    # predict = clf.predict(test_data)
    # print(accuracy_score(test_target, predict))
    # print(cohen_kappa_score(test_target, predict))
