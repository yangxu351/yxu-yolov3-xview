import random
import numpy as np
import random_test2

def init_seed():
    random.seed(0)

def random_mosaic():
    print(random.random())

def random_affine():
    print(random.random())
    random_test2.random_mosaic()


if __name__ == '__main__':
    random_test2.init_seed()
    random_affine()
