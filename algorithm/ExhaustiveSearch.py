from tqdm import tqdm
from deap import base
from deap import creator
import pandas as pd
import numpy

from code import Params
from evaluation_function.Fitness import Fitness

class ExhaustiveSearch():

    def run(self):
        try:
            sequence = pd.read_csv("sequences3.csv", names=['keys', 'values'])
            sequencelog = {}
            sequencelog.update(dict(zip(sequence['keys'], sequence['values'])))
        except:
            sequencelog = {}
        ev = Fitness()
        for a in tqdm(range(Params.N_ALGORITHMS +1)):
            for b in range(Params.N_ALGORITHMS +1):
                for c in range(Params.N_ALGORITHMS +1):
                    for d in range(Params.N_ALGORITHMS +1):
                        #for e in range(Params.N_ALGORITHMS):
                        seq = list(filter(lambda x: x != 0, [a,b,c,d]))
                        if(len(seq) > 0):
                            if (str(seq) not in sequencelog):
                                fit = ev.evaluate(seq)
                                sequencelog[str(seq)] = numpy.float(fit[0])
                            pd.DataFrame.from_dict(sequencelog, orient='index').to_csv("sequences3.csv")
