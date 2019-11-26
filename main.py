import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from core import Core
from nn import NN

def main():
    test3 = Core()
    test3.loadData('../inversion/DATA/SQUAREDUCT/DATA/03500_full.csv')
    test3.calculateGradient()
    
    data = test3.data
    
main()
