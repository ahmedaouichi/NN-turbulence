import numpy as np
from core import Core
from nn import NN


def main():
    test1 = NN()
    test2 = Core(10)
    test2.calc()
    print(test1.name," and ",test2.x)


main()
