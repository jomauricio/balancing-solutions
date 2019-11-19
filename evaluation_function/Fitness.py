import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import OneSidedSelection
from imblearn.combine import SMOTETomek
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from imblearn.metrics import geometric_mean_score

from code import Params

class Fitness():

    def __init__(self):
        self.problem_type = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        self.n = 5          # media de n execuções

    def evaluate(self, individual):
        if(len(individual) > 0):
            test = Params.BASE_TEST
            # train = Params.BASE_TRAIN



            # test_target = test['X21'].tolist()
            # test_data = test.drop(['X21'], 1)
            # test_data, test_target = test.iloc[:,1:52], test["classe"]
            # target = test['Class'].tolist()
            # target = test['Class'].map(lambda x: 1 if x == 'positive' else 0).values
            # data = test.drop(['Class'], 1)

            target = test['X37'].tolist()
            data = test.drop(['X37'], 1)

            train_data, test_data, train_target, test_target = train_test_split(data, target, random_state=42, stratify=target)

            # train_target = train['classe'].tolist()
            # train = train.drop(['classe'], 1)
            # train_data = train.as_matrix()
            # train_data, train_target = train.iloc[:,1:52], train["classe"]

            acc = 0
            fscore =0
            gm = 0
            recall = 0
            precision = 0
            auc = 0
            for i in range(self.n):
                output_train_data = train_data
                output_train_target = train_target
                try:
                    for feature in individual:
                        if (feature == 1):
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


                    clf = RandomForestClassifier(n_estimators=10, max_features=6, warm_start=True, oob_score=True)
                    clf.fit(output_train_data, output_train_target)
                    predict = clf.predict(test_data)

                    acc += accuracy_score(test_target, predict)
                    fscore += f1_score(test_target, predict)
                    gm += geometric_mean_score(test_target, predict)
                    recall += recall_score(test_target, predict)
                    precision += precision_score(test_target, predict)
                    auc += roc_auc_score(test_target, predict)
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
            acc = acc/self.n
            fscore = fscore/self.n
            gm = gm/self.n
            recall = recall/self.n
            precision = precision/self.n
            auc = auc/self.n
        else:
            acc = -1
            fscore = -1
            gm = -1
            recall = -1
            precision = -1
            auc = -1
        return acc, fscore, gm, recall, precision, auc