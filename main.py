import numpy as np
from core import Calculator
from network import NN


def main():
    test1 = NN()
    test2 = Calculator(10)
    test2.calc()
    print(test1.name," and ",test2.x)


main()
