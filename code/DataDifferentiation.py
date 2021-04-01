import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import TomekLinks

from code import Params


class DataDifferentiation:

    def evaluate(self, individual):
        test = Params.BASE_TEST
        # train = Params.BASE_TRAIN

        # test_target = test['X21'].tolist()
        # test = test.drop(['X21'], 1)
        # test_data = test.as_matrix()
        # test_data, test_target = test.iloc[:,1:52], test["classe"]
        target = test['X37'].tolist()
        data = test.drop(['X37'], 1)
        # train_data, test_data, train_target, test_target = train_test_split(data, target, random_state=42,
        # stratify=target)

        # train_target = train['classe'].tolist()
        # train = train.drop(['classe'], 1)
        # train_data = train.as_matrix()
        # train_data, train_target = train.iloc[:,1:52], train["classe"]

        output_train_data = data
        output_train_target = target
        df = pd.DataFrame(data=output_train_data, columns=data.columns)
        df['classe'] = output_train_target
        n = 0
        try:
            for feature in individual:
                n = n + 1
                df_ant = df
                if feature == 1:
                    nm = NearMiss()
                    output_train_data, output_train_target = nm.fit_resample(output_train_data, output_train_target)

                    df = pd.DataFrame(data=output_train_data, columns=data.columns)
                    df['classe'] = output_train_target
                    indc = list(nm.sample_indices_)
                    dif = df_ant.drop(indc)
                    if not dif.empty:
                        dif.to_csv("Diferença entre os passos " + str(n) + "e " + str(n - 1) + ".csv")
                elif feature == 2:
                    enn = EditedNearestNeighbours()
                    output_train_data, output_train_target = enn.fit_resample(output_train_data, output_train_target)

                    df = pd.DataFrame(data=output_train_data, columns=data.columns)
                    df['classe'] = output_train_target
                    indc = list(enn.sample_indices_)
                    dif = df_ant.drop(indc)
                    if not dif.empty:
                        dif.to_csv("Diferença entre os passos " + str(n) + "e " + str(n - 1) + ".csv")
                elif feature == 3:
                    output_train_data, output_train_target = SMOTE().fit_resample(output_train_data,
                                                                                  output_train_target)

                    df = pd.DataFrame(data=output_train_data, columns=data.columns)
                    df['classe'] = output_train_target
                    indc = list(df_ant.index)
                    dif = df.drop(indc)
                    if not dif.empty:
                        dif.to_csv("Diferença entre os passos " + str(n) + "e " + str(n - 1) + ".csv")
                elif feature == 4:
                    tl = TomekLinks()
                    output_train_data, output_train_target = tl.fit_resample(output_train_data, output_train_target)

                    df = pd.DataFrame(data=output_train_data, columns=data.columns)
                    df['classe'] = output_train_target

                    indc = list(tl.sample_indices_)
                    dif = df_ant.drop(indc)
                    if not dif.empty:
                        dif.to_csv("Diferença entre os passos " + str(n) + "e " + str(n - 1) + ".csv")
                elif feature == 5:
                    oss = OneSidedSelection()
                    output_train_data, output_train_target = oss.fit_resample(output_train_data, output_train_target)

                    df = pd.DataFrame(data=output_train_data, columns=data.columns)
                    df['classe'] = output_train_target

                    indc = list(oss.sample_indices_)
                    dif = df_ant.drop(indc)
                    if not dif.empty:
                        dif.to_csv("Diferença entre os passos " + str(n) + "e " + str(n - 1) + ".csv")
        except:
            print("collision")
