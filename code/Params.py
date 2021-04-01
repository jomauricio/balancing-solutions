"""
Configuration for the problem
"""

import pandas as pd

INDIVIDUAL_SIZE = 5
N_ALGORITHMS = 5

BASE_TEST = pd.read_csv("data/satimage.csv")

POPULATION_SIZE = 10
NGENERATIONS = 10