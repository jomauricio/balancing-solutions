from code import Params

import numpy as np
# from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import (EditedNearestNeighbours, NearMiss,
                                     OneSidedSelection, TomekLinks)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)


class Fitness:

    def __init__(self):
        self.problem_type = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        self.n = Params.N_EXECUTIONS  # media de n execuções

    def evaluate(self, individual):
        if len(individual) > 0:
            test = Params.BASE_TEST
            # train = Params.BASE_TRAIN

            # test_target = test['X21'].tolist()
            # test_data = test.drop(['X21'], 1)
            # test_data, test_target = test.iloc[:,1:52], test["classe"]
            # target = test['Class'].tolist()
            # target = test['Class'].map(lambda x: 1 if x == 'positive' else 0).values
            # data = test.drop(['Class'], 1)

            target = test['Result'].tolist()
            data = test.drop(['Result'], 1)

            # train_data, test_data, train_target, test_target = train_test_split(data, target, random_state=42, stratify = target)
            name_metrics = ['accuracy', 'f1_macro']

            # train_target = train['classe'].tolist()
            # train = train.drop(['classe'], 1)
            # train_data = train.as_matrix()
            # train_data, train_target = train.iloc[:,1:52], train["classe"]

            acc = 0
            fscore = 0
            gm = 0
            recall = 0
            precision = 0
            auc = 0
            for i in range(self.n):
                output_data = data
                output_target = target
                try:
                    for feature in individual:
                        if (feature == 1):
                            output_data, output_target = NearMiss().fit_resample(output_data,
                                                                                 output_target)
                        elif (feature == 2):
                            output_data, output_target = EditedNearestNeighbours().fit_resample(
                                output_data, output_target)
                        elif (feature == 3):
                            output_data, output_target = SMOTE().fit_resample(output_data,
                                                                              output_target)
                        elif (feature == 4):
                            output_data, output_target = TomekLinks().fit_resample(output_data,
                                                                                   output_target)
                        elif (feature == 5):
                            output_data, output_target = OneSidedSelection().fit_resample(output_data,
                                                                                          output_target)
                    stratifiedKFold = StratifiedKFold(
                        n_splits=10, shuffle=True, random_state=i)

                    if(Params.CLASSIFIER == 1):
                        clf = RandomForestClassifier(
                            n_estimators=10, max_features=6, warm_start=True, oob_score=True)
                    elif (Params.CLASSIFIER == 2):
                        clf = KNeighborsClassifier(n_neighbors=3)
                    elif(Params.CLASSIFIER == 3):
                        clf = make_pipeline(
                            StandardScaler(), SVC(gamma='auto'))
                    else:
                        clf = RandomForestClassifier(
                            n_estimators=10, max_features=6, warm_start=True, oob_score=True)

                    # clf.fit(output_data, output_target)
                    # predict = clf.predict(test_data)

                    # clf = KNeighborsClassifier(n_neighbors=3)
                    # clf.fit(output_data, output_target)
                    # predict = clf.predict(test_data)

                    # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                    # clf.fit(output_data, output_target)
                    # predict = clf.predict(test_data)

                    metrics = cross_validate(
                        clf, output_data, output_target, cv=stratifiedKFold, scoring=name_metrics)

                    acc += np.mean(metrics['accuracy'])
                    fscore += np.mean(metrics['f1'])

                    # acc += accuracy_score(test_target, predict)
                    # fscore += f1_score(test_target, predict)
                    # gm += geometric_mean_score(test_target, predict)
                    # recall += recall_score(test_target, predict)
                    # precision += precision_score(test_target, predict)
                    # auc += roc_auc_score(test_target, predict)
                    # l_acc.append(fit)
                    # score = pipeline_a.decision_function(X_test)
                    # kappa = cohen_kappa_score(test_target, predict)
                except:
                    acc = 0
                    fscore = 0
                    gm = 0
                    recall = 0
                    precision = 0
                    auc = 0
                    return acc, fscore, gm, recall, precision, auc
            acc = acc / self.n
            fscore = fscore / self.n
            gm = gm / self.n
            recall = recall / self.n
            precision = precision / self.n
            auc = auc / self.n
        else:
            acc = -1
            fscore = -1
            gm = -1
            recall = -1
            precision = -1
            auc = -1
        return acc, fscore, gm, recall, precision, auc
