"""
Configuration for the problem
"""

import pandas as pd

INDIVIDUAL_SIZE = 5
N_ALGORITHMS = 5
N_EXECUTIONS = 5
CLASSIFIER = 1  # 1 - RF, 2 - KN, 3 - SVN

BASE_TEST = pd.read_csv("code/data/Training Dataset-UCI.csv")

POPULATION_SIZE = 3
NGENERATIONS = 3
