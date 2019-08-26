'''
Configuration for the problem
'''

import pandas as pd

INDIVIDUAL_SIZE = 5
N_ALGORITHMS = 6

BASE_TRAIN = pd.read_csv("pt - pt.csv")
BASE_TEST = pd.read_csv("en - en.csv")

POPULATION_SIZE = 100
NGENERATIONS = 1000