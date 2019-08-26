import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import OneSidedSelection
from imblearn.combine import SMOTETomek

from code import Params

class Fitness():

    def __init__(self):
        self.problem_type = (1.0,)

    def evaluate(self, individual):
        if(len(individual) > 2):
            test = Params.BASE_TEST
            train = Params.BASE_TRAIN

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
                acc = accuracy_score(test_target, predict)
                # fscore = f1_score(y_test, predict, average="weighted")
                # gm = geometric_mean_score(y_test, predict, average="weighted")
                # score = pipeline_a.decision_function(X_test)
                # auroc = roc_auc_score(y_test, score, average="weighted")
                # kappa = cohen_kappa_score(test_target, predict)
            except:
                acc = 0
                # kappa = 0
                # fscore = 0
                # gm = 0
                # auroc = 0
            #return acc,# fscore, gm #, auroc,
        else:
            acc = 0
        return acc,