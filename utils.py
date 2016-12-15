#!/usr/bin/env python
"""
utils.py
- Utility functions for the project

Author: Long Qian
Date: Dec 2016

"""
import numpy as np


def getRandomRotation( ):
    """
    Get a random Rotation matrix
    :return: A numpy 3*3 array
    """
    M = np.random.rand(3, 3)
    Q, R = np.linalg.qr(M)
    return Q.astype(np.float32)


def getRandomTranslation( ):
    """
    Get a random translation vector
    :return: A numpy 3-element array, within range 0-1
    """
    T = np.random.rand(3)
    return T.astype(np.float32)



if __name__ == '__main__':
    """
    When utils.py is used as main module
    - Demonstrate function getRandomRotation()
    - Demonstrate function getRandomTranslation()
    """
    print 'Random Rotation:'
    print getRandomRotation()
    print 'Random Translation:'
    print getRandomTranslation()
